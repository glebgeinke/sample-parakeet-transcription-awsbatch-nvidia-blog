#!/usr/bin/env python3
"""
Parakeet Long-Form Audio Transcription Script

Supports two inference modes:
  - Simple (default): Uses NeMo's built-in model.transcribe() for straightforward processing.
  - Streaming: Buffered streaming inference that processes audio in overlapping chunks,
    reducing peak VRAM usage for very long audio files.

Streaming mode is enabled via the STREAMING_ENABLED environment variable or the --streaming
CLI flag. Chunk parameters are configurable via CHUNK_SECS, LEFT_CONTEXT_SECS, and
RIGHT_CONTEXT_SECS environment variables (or equivalent CLI flags).

Usage:
    python3 parakeet_transcribe.py input_bucket_name input_file_key [--output OUTPUT] [--timestamps] [--streaming] [--verbose]
Input:
    input_bucket_name = sys.argv[1]
    input_file_key = sys.argv[2]
Output:
    output_bucket_name = os.environ['OUTPUT_BUCKET_NAME']
    output_file_prefix = os.environ['OUTPUT_FILE_PREFIX']

If input_bucket_name and input_file_key are provided, input file will be downloaded from S3. Output transcript will be uploaded to S3.
If not, script will work with local files.

Requires boto3 for S3 usage.
"""

import argparse
import copy
import gc
import io
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torchaudio
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, open_dict
from tqdm.auto import tqdm

from pydub import AudioSegment
import librosa
import soundfile as sf

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from nemo.collections.asr.models import ASRModel, EncDecRNNTModel, EncDecHybridRNNTCTCModel
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
    from nemo.collections.asr.parts.utils.streaming_utils import (
        ContextSize,
        StreamingBatchedAudioBuffer,
    )
    from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs
except ImportError as e:
    print(f"Error importing NeMo: {e}")
    print("Please install NeMo toolkit: pip install nemo_toolkit[asr]")
    sys.exit(1)

class ParakeetTranscriber:
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "auto",
        batch_size: int = 4,
        verbose: bool = False,
        streaming: bool = False,
        chunk_secs: float = 20.0,
        left_context_secs: float = 5.0,
        right_context_secs: float = 3.0
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sample_rate = 16000
        self.mono_channel = True
        self.attention_model = "rel_pos_local_attn"
        self.attention_window = [256, 256]
        self.subsampling_factor = 1
        self.precision = torch.bfloat16
        self.streaming = streaming
        self.chunk_secs = chunk_secs
        self.left_context_secs = left_context_secs
        self.right_context_secs = right_context_secs
        self._setup_logging()
        self._load_model()

    def _setup_logging(self):
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        self.logger.info(f"Loading Parakeet model: {self.model_name}")
        self.model = ASRModel.from_pretrained(model_name=self.model_name)
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        if self.streaming:
            self._setup_streaming_model()
        else:
            self.model.change_attention_model(self.attention_model, self.attention_window)
            self.model.change_subsampling_conv_chunking_factor(self.subsampling_factor)
        if self.device == "cuda":
            self.model.to(self.precision)
            self.logger.info(f"Model precision set to: {self.precision}")

    def _setup_streaming_model(self):
        """Configure model for buffered streaming inference."""
        self.model.freeze()

        model_cfg = copy.deepcopy(self.model._cfg)
        OmegaConf.set_struct(model_cfg.preprocessor, False)
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0
        if model_cfg.preprocessor.normalize != "per_feature":
            self.logger.error(
                "Only models trained with per_feature normalization are supported for streaming"
            )
        OmegaConf.set_struct(model_cfg.preprocessor, True)

        # Configure decoding for streaming (greedy_batch with label-looping)
        decoding_cfg = OmegaConf.structured(RNNTDecodingConfig())
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy_batch"
            decoding_cfg.greedy.loop_labels = True
            decoding_cfg.tdt_include_token_duration = True
            decoding_cfg.greedy.preserve_alignments = False
            decoding_cfg.fused_batch_size = -1
            decoding_cfg.beam.return_best_hypothesis = True

        if isinstance(self.model, EncDecRNNTModel):
            self.model.change_decoding_strategy(decoding_cfg)
        elif hasattr(self.model, 'cur_decoder'):
            self.model.change_decoding_strategy(decoding_cfg, decoder_type='rnnt')

        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0

        # Pre-compute streaming context sizes
        audio_sample_rate = model_cfg.preprocessor['sample_rate']
        feature_stride_sec = model_cfg.preprocessor['window_stride']
        features_per_sec = 1.0 / feature_stride_sec
        encoder_subsampling_factor = self.model.encoder.subsampling_factor

        features_frame2audio_samples = self._make_divisible_by(
            int(audio_sample_rate * feature_stride_sec),
            factor=encoder_subsampling_factor
        )
        encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

        context_encoder_frames = ContextSize(
            left=int(self.left_context_secs * features_per_sec / encoder_subsampling_factor),
            chunk=int(self.chunk_secs * features_per_sec / encoder_subsampling_factor),
            right=int(self.right_context_secs * features_per_sec / encoder_subsampling_factor),
        )
        context_samples = ContextSize(
            left=context_encoder_frames.left * encoder_subsampling_factor * features_frame2audio_samples,
            chunk=context_encoder_frames.chunk * encoder_subsampling_factor * features_frame2audio_samples,
            right=context_encoder_frames.right * encoder_subsampling_factor * features_frame2audio_samples,
        )

        self._audio_sample_rate = audio_sample_rate
        self._encoder_frame2audio_samples = encoder_frame2audio_samples
        self._context_samples = context_samples

        self.logger.info(
            f"Streaming contexts (sec): "
            f"Left {context_samples.left / audio_sample_rate:.2f}, "
            f"Chunk {context_samples.chunk / audio_sample_rate:.2f}, "
            f"Right {context_samples.right / audio_sample_rate:.2f}"
        )
        latency_secs = (context_samples.chunk + context_samples.right) / audio_sample_rate
        self.logger.info(f"Theoretical latency: {latency_secs:.2f}s")

    @staticmethod
    def _make_divisible_by(num: int, factor: int) -> int:
        return (num // factor) * factor

    def _validate_audio_file(self, audio_path: str) -> bool:
        try:
            info = sf.info(audio_path)
            self.logger.debug(f"Audio info: {info.duration:.2f}s, {info.samplerate}Hz, {info.channels} channels")
            return True
        except Exception as e:
            self.logger.error(f"Invalid audio file {audio_path}: {e}")
            return False

    def _convert_mp3_with_librosa(self, mp3_path: str) -> Tuple[str, float]:
        self.logger.info("Converting MP3 to WAV using librosa...")
        audio_data, sr = librosa.load(mp3_path, sr=self.target_sample_rate, mono=self.mono_channel)
        duration_sec = len(audio_data) / sr
        wav_path = str(Path(mp3_path).with_name(Path(mp3_path).stem + '_converted.wav'))
        sf.write(wav_path, audio_data, sr)
        return wav_path, duration_sec

    def _preprocess_audio(self, audio_path: str) -> Tuple[str, float]:
        self.logger.info(f"Processing audio: {Path(audio_path).name}")
        if audio_path.lower().endswith('.mp3'):
            return self._convert_mp3_with_librosa(audio_path)
        processed_path = str(Path(audio_path).stem) + '_processed.wav'
        if audio_path.lower().endswith('.wav'):
            waveform, sample_rate = torchaudio.load(audio_path)
            duration_sec = waveform.shape[1] / sample_rate
            if self.mono_channel and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
            torchaudio.save(processed_path, waveform, self.target_sample_rate)
            return processed_path, duration_sec
        # fallback
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds
        if self.mono_channel and audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != self.target_sample_rate:
            audio = audio.set_frame_rate(self.target_sample_rate)
        audio.export(processed_path, format="wav")
        return processed_path, duration_sec

    def transcribe(
        self,
        audio_path: str,
        output_file: Optional[str] = None,
        save_timestamps: bool = False,
        cleanup_temp: bool = True
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        processed_path = None
        try:
            if not self._validate_audio_file(audio_path):
                raise ValueError(f"Invalid audio file: {audio_path}")
            processed_path, duration_sec = self._preprocess_audio(audio_path)
            self.logger.info(f"Starting transcription (duration: {duration_sec:.2f}s)...")
            start_time = time.time()

            if self.streaming:
                transcription_text, timestamps_df = self._transcribe_streaming(
                    processed_path, save_timestamps
                )
            else:
                transcription_text, timestamps_df = self._transcribe_simple(
                    processed_path, save_timestamps
                )

            processing_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription_text)
                self.logger.info(f"Transcription saved to: {output_file}")
                if timestamps_df is not None:
                    timestamp_file = Path(output_file).with_suffix('.csv')
                    timestamps_df.to_csv(timestamp_file, index=False)
                    self.logger.info(f"Timestamps saved to: {timestamp_file}")

            return transcription_text, timestamps_df
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
        finally:
            if cleanup_temp and processed_path and processed_path != audio_path and os.path.exists(processed_path):
                os.remove(processed_path)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _transcribe_simple(
        self,
        processed_path: str,
        save_timestamps: bool
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """Transcribe using NeMo's built-in model.transcribe() method."""
        if save_timestamps:
            output = self.model.transcribe([processed_path], timestamps=True)
        else:
            output = self.model.transcribe([processed_path])
        if not output or not isinstance(output, list) or not output[0]:
            self.logger.warning("Empty transcription output")
            return "", None
        transcription_text = output[0].text if hasattr(output[0], 'text') else ""
        timestamps_df = None
        if save_timestamps and hasattr(output[0], 'timestamp') and 'segment' in output[0].timestamp:
            try:
                segments = output[0].timestamp['segment']
                timestamps_data = {
                    "Start (s)": [round(ts['start'], 2) for ts in segments],
                    "End (s)": [round(ts['end'], 2) for ts in segments],
                    "Segment": [ts['segment'] for ts in segments]
                }
                timestamps_df = pd.DataFrame(timestamps_data)
                self.logger.info(f"Extracted {len(timestamps_df)} timestamp segments")
            except Exception as e:
                self.logger.warning(f"Failed to extract timestamps: {e}")
        return transcription_text, timestamps_df

    def _transcribe_streaming(
        self,
        processed_path: str,
        save_timestamps: bool
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """Transcribe using buffered streaming inference with overlapping chunks.

        Processes audio in [left_context + chunk + right_context] windows, decoding
        only the chunk portion at each step. This keeps VRAM usage bounded regardless
        of total audio length.
        """
        audio_array, _ = librosa.load(processed_path, sr=self._audio_sample_rate)
        audio_tensor = torch.tensor(audio_array).unsqueeze(0)
        audio_length = torch.tensor([len(audio_array)])
        self.logger.info(
            f"Audio loaded for streaming. Duration: {len(audio_array) / self._audio_sample_rate:.2f}s"
        )

        decoding_computer = self.model.decoding.decoding.decoding_computer
        all_hyps = []

        with torch.no_grad(), torch.inference_mode():
            audio_batch = audio_tensor.to(device=self.device)
            audio_batch_lengths = audio_length.to(device=self.device)
            batch_size = audio_batch.shape[0]
            device = audio_batch.device

            current_batched_hyps = None
            state = None
            left_sample = 0
            right_sample = min(
                self._context_samples.chunk + self._context_samples.right,
                audio_batch.shape[1]
            )
            buffer = StreamingBatchedAudioBuffer(
                batch_size=batch_size,
                context_samples=self._context_samples,
                dtype=audio_batch.dtype,
                device=device,
            )
            rest_audio_lengths = audio_batch_lengths.clone()

            pbar = tqdm(total=audio_batch.shape[1], desc="Streaming inference", unit="samples")
            while left_sample < audio_batch.shape[1]:
                chunk_length = min(right_sample, audio_batch.shape[1]) - left_sample
                is_last_chunk_batch = chunk_length >= rest_audio_lengths
                is_last_chunk = right_sample >= audio_batch.shape[1]
                chunk_lengths_batch = torch.where(
                    is_last_chunk_batch,
                    rest_audio_lengths,
                    torch.full_like(rest_audio_lengths, fill_value=chunk_length),
                )
                buffer.add_audio_batch_(
                    audio_batch[:, left_sample:right_sample],
                    audio_lengths=chunk_lengths_batch,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=is_last_chunk_batch,
                )

                encoder_output, encoder_output_len = self.model(
                    input_signal=buffer.samples,
                    input_signal_length=buffer.context_size_batch.total(),
                )
                encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
                encoder_context = buffer.context_size.subsample(
                    factor=self._encoder_frame2audio_samples
                )
                encoder_context_batch = buffer.context_size_batch.subsample(
                    factor=self._encoder_frame2audio_samples
                )
                encoder_output = encoder_output[:, encoder_context.left:]

                chunk_batched_hyps, _, state = decoding_computer(
                    x=encoder_output,
                    out_len=encoder_context_batch.chunk,
                    prev_batched_state=state,
                )
                if current_batched_hyps is None:
                    current_batched_hyps = chunk_batched_hyps
                else:
                    current_batched_hyps.merge_(chunk_batched_hyps)

                rest_audio_lengths -= chunk_lengths_batch
                left_sample = right_sample
                pbar.update(self._context_samples.chunk)
                right_sample = min(
                    right_sample + self._context_samples.chunk,
                    audio_batch.shape[1]
                )

            pbar.close()
            all_hyps.extend(
                batched_hyps_to_hypotheses(current_batched_hyps, None, batch_size=batch_size)
            )

        # Post-process hypotheses
        for i, hyp in enumerate(all_hyps):
            hyp.text = self.model.tokenizer.ids_to_text(hyp.y_sequence.tolist())
            if save_timestamps:
                hyp = self.model.decoding.compute_rnnt_timestamps(hyp)
                hyp = process_timestamp_outputs(
                    hyp,
                    subsampling_factor=self.model.encoder.subsampling_factor,
                    window_stride=self.model.cfg['preprocessor']['window_stride'],
                )
                all_hyps[i] = hyp

        transcription_text = all_hyps[0].text if all_hyps else ""
        timestamps_df = None
        if save_timestamps and all_hyps and hasattr(all_hyps[0], 'timestamp'):
            try:
                timestamps_data = all_hyps[0].timestamp
                if timestamps_data and isinstance(timestamps_data, dict):
                    segments = timestamps_data.get('segment', [])
                    if segments:
                        timestamps_df = pd.DataFrame({
                            "Start (s)": [round(ts['start'], 2) for ts in segments],
                            "End (s)": [round(ts['end'], 2) for ts in segments],
                            "Segment": [ts['segment'] for ts in segments]
                        })
                        self.logger.info(f"Extracted {len(timestamps_df)} timestamp segments")
            except Exception as e:
                self.logger.warning(f"Failed to extract timestamps: {e}")
        return transcription_text, timestamps_df

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio using Parakeet long-form approach")
    parser.add_argument("input_bucket_name", nargs="?", help="Input S3 bucket name (optional for local files)")
    parser.add_argument("input_file_key", nargs="?", help="Input S3 file key (optional for local files)")
    parser.add_argument("--output", "-o", help="Output file for transcription (default: transcript.txt)")
    parser.add_argument("--timestamps", "-t", action="store_true", help="Extract timestamps and save to CSV")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size for processing (default: 4)")
    parser.add_argument("--device", "-d", choices=["auto", "cuda", "cpu"], default="auto", help="Device to use for inference (default: auto)")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3", help="Parakeet model identifier (default: nvidia/parakeet-tdt-0.6b-v3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--version", action="version", version="Parakeet Transcriber 1.1")
    parser.add_argument(
        "--streaming", action="store_true",
        default=os.environ.get('STREAMING_ENABLED', '').lower() in ('true', '1', 'yes'),
        help="Enable buffered streaming inference (env: STREAMING_ENABLED)"
    )
    parser.add_argument(
        "--chunk-secs", type=float,
        default=float(os.environ.get('CHUNK_SECS', '20')),
        help="Chunk length in seconds for streaming (env: CHUNK_SECS, default: 20)"
    )
    parser.add_argument(
        "--left-context-secs", type=float,
        default=float(os.environ.get('LEFT_CONTEXT_SECS', '5.0')),
        help="Left context in seconds for streaming (env: LEFT_CONTEXT_SECS, default: 5.0)"
    )
    parser.add_argument(
        "--right-context-secs", type=float,
        default=float(os.environ.get('RIGHT_CONTEXT_SECS', '3.0')),
        help="Right context in seconds for streaming (env: RIGHT_CONTEXT_SECS, default: 3.0)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    s3_mode = args.input_bucket_name and args.input_file_key

    s3_client = None
    local_audio_path = None

    if s3_mode:
        if not S3_AVAILABLE:
            print("boto3 is required for S3 input.")
            sys.exit(1)
        input_bucket_name = args.input_bucket_name
        input_file_key = args.input_file_key
        output_bucket_name = os.environ['OUTPUT_BUCKET_NAME']
        output_file_prefix = os.environ['OUTPUT_FILE_PREFIX']
        s3_client = boto3.client('s3')
        local_audio_path = input_file_key.split('/')[-1]
        print(f"Downloading input file from S3: s3://{input_bucket_name}/{input_file_key} -> {local_audio_path}")
        s3_client.download_file(input_bucket_name, input_file_key, local_audio_path)
        output_file = local_audio_path + ".txt"
    else:
        if not args.output:
            print("No output file specified, will save to transcript.txt")
            args.output = "transcript.txt"
        local_audio_path = args.input_bucket_name if args.input_bucket_name else ""
        output_file = args.output

    transcriber = ParakeetTranscriber(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        verbose=args.verbose,
        streaming=args.streaming,
        chunk_secs=args.chunk_secs,
        left_context_secs=args.left_context_secs,
        right_context_secs=args.right_context_secs,
    )

    print(f"Transcribing: {local_audio_path}")
    transcription, timestamps = transcriber.transcribe(
        audio_path=local_audio_path,
        output_file=output_file,
        save_timestamps=args.timestamps
    )

    print("TRANSCRIPTION COMPLETE!")
    print(f"Transcription saved to: {output_file}")
    print(f"Text length: {len(transcription)} characters")
    print(f"Word count: {len(transcription.split())}")
    if timestamps is not None:
        timestamp_file = Path(output_file).with_suffix('.csv')
        print(f"Timestamps saved to: {timestamp_file}")
        print(f"Timestamp segments: {len(timestamps)}")
    preview = transcription[:200]
    print(f"\nPreview (first 200 chars):\n{'-'*50}\n{preview + ('...' if len(transcription)>200 else '')}\n{'-'*50}")

    if s3_mode:
        s3_output_key = output_file_prefix + output_file
        print(f"Uploading transcript to S3: s3://{output_bucket_name}/{s3_output_key}")
        s3_client.upload_file(output_file, output_bucket_name, s3_output_key)
        if timestamps is not None:
            s3_csv_key = output_file_prefix + Path(output_file).with_suffix('.csv').name
            print(f"Uploading timestamps to S3: s3://{output_bucket_name}/{s3_csv_key}")
            s3_client.upload_file(str(Path(output_file).with_suffix('.csv')), output_bucket_name, s3_csv_key)
        if os.path.exists(local_audio_path): os.remove(local_audio_path)
        if os.path.exists(output_file): os.remove(output_file)
        if timestamps is not None and os.path.exists(Path(output_file).with_suffix('.csv')): os.remove(Path(output_file).with_suffix('.csv'))

if __name__ == "__main__":
    main()
