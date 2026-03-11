"""Main entry point for yt-transcriber CLI."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .cli import parse_args
from .config import Config
from .downloader import download_audio, cleanup_audio_file, DownloadError
from .file_writer import write_transcript, combine_transcripts
from .playlist import process_playlist_with_retry, PlaylistError
from .transcript import get_official_transcript, NoTranscriptError
from .url_parser import parse_input, read_urls_from_file, extract_video_id
from .whisper import transcribe_with_whisper, get_device
from .url_parser import extract_video_id as url_extract_video_id


class Transcriber:
    """Main transcription engine."""

    def __init__(self, config: Config):
        self.config = config
        self.successful = []
        self.failed = []

    async def transcribe_video(
        self, video_id: str, index: Optional[int] = None, total: Optional[int] = None
    ) -> Optional[str]:
        """
        Transcribe a single YouTube video.

        Args:
            video_id: YouTube video ID
            index: Current video index (for progress)
            total: Total videos (for progress)

        Returns:
            Transcript text or None if failed
        """
        prefix = ""
        if index is not None and total is not None:
            prefix = f"[{index}/{total}] "

        # Step 1: Try official transcript
        if self.config.verbose:
            print(f"{prefix}Processing video {video_id}...")

        try:
            transcript, language = get_official_transcript(video_id, language=self.config.language)

            if self.config.verbose:
                print(f"{prefix}✓ Using official transcript ({language})")

            # Write transcript
            title = f"Video {video_id}"  # Default title
            file_path = write_transcript(
                video_id=video_id,
                title=title,
                transcript=transcript,
                output_dir=self.config.output_dir,
                source="official",
                language=language,
            )

            self.successful.append(
                {
                    "video_id": video_id,
                    "file_path": file_path,
                    "source": "official",
                }
            )

            return transcript

        except NoTranscriptError:
            if self.config.verbose:
                print(f"{prefix}✗ No official transcript, using Whisper...")

            # Step 2: Fallback to Whisper transcription
            try:
                # Download audio
                audio_path = download_audio(video_id, self.config.output_dir)

                try:
                    # Transcribe with Whisper
                    if self.config.verbose:
                        print(
                            f"{prefix}  → Transcribing with Whisper ({self.config.model_size}, {get_device()})..."
                        )

                    transcript = transcribe_with_whisper(
                        audio_path=audio_path,
                        model_size=self.config.model_size,
                        language=self.config.language,
                        device=self.config.device,
                    )

                    if self.config.verbose:
                        print(f"{prefix}  ✓ Whisper transcription complete")

                    # Write transcript
                    title = f"Video {video_id}"  # Default title
                    file_path = write_transcript(
                        video_id=video_id,
                        title=title,
                        transcript=transcript,
                        output_dir=self.config.output_dir,
                        source="whisper",
                        language=self.config.language or "auto",
                    )

                    self.successful.append(
                        {
                            "video_id": video_id,
                            "file_path": file_path,
                            "source": "whisper",
                        }
                    )

                    return transcript

                finally:
                    # Cleanup audio file
                    cleanup_audio_file(audio_path)

            except DownloadError as e:
                if self.config.verbose:
                    print(f"{prefix}  ✗ Download failed: {e}")
                self.failed.append(
                    {
                        "video_id": video_id,
                        "reason": f"Download error: {e}",
                    }
                )
                return None

            except Exception as e:
                if self.config.verbose:
                    print(f"{prefix}  ✗ Whisper transcription failed: {e}")
                self.failed.append(
                    {
                        "video_id": video_id,
                        "reason": f"Whisper error: {e}",
                    }
                )
                return None

        except Exception as e:
            if self.config.verbose:
                print(f"{prefix}  ✗ Error: {e}")
            self.failed.append(
                {
                    "video_id": video_id,
                    "reason": f"Unexpected error: {e}",
                }
            )
            return None

    async def process_single_video(self, video_id: str) -> Optional[str]:
        """Process a single video."""
        return await self.transcribe_video(video_id)

    async def process_playlist(self, playlist_url: str) -> None:
        """Process all videos in a playlist."""
        if self.config.verbose:
            print(f"Processing playlist: {playlist_url}")

        try:
            successful, failed = await process_playlist_with_retry(
                playlist_url=playlist_url,
                process_video_func=self.transcribe_video,
                sleep_delay=self.config.sleep_delay,
            )

            self.successful.extend(successful)
            self.failed.extend(failed)

        except PlaylistError as e:
            print(f"Error processing playlist: {e}")
            sys.exit(1)

    async def process_file(self, file_path: str) -> None:
        """Process URLs from a file."""
        if self.config.verbose:
            print(f"Reading URLs from file: {file_path}")

        try:
            urls = read_urls_from_file(file_path)
            if self.config.verbose:
                print(f"Found {len(urls)} URLs to process")

            # Process each URL
            for index, url in enumerate(urls, 1):
                video_id = extract_video_id(url)
                if video_id:
                    await self.transcribe_video(video_id, index, len(urls))
                else:
                    self.failed.append(
                        {
                            "video_id": "unknown",
                            "reason": f"Invalid URL: {url}",
                        }
                    )

        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

    def create_combined_file(self) -> None:
        """Create combined transcription file."""
        if not self.successful:
            return

        file_paths = [s["file_path"] for s in self.successful]
        combined_path = combine_transcripts(file_paths, self.config.output_dir, "combined.txt")

        if self.config.verbose:
            print(f"\nCombined file created: {combined_path}")

    def print_summary(self) -> None:
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)

        total = len(self.successful) + len(self.failed)

        print(f"Total videos: {total}")
        print(f"✓ Successful: {len(self.successful)}")
        print(f"✗ Failed: {len(self.failed)}")

        if self.successful:
            print("\nSuccessful videos:")
            for item in self.successful:
                source_icon = "✓" if item["source"] == "official" else "⚡"
                print(f"  {source_icon} {item['video_id']} ({item['source']})")

        if self.failed:
            print("\nFailed videos:")
            for item in self.failed:
                print(f"  ✗ {item['video_id']}: {item['reason']}")

        if self.config.combine and self.successful:
            print(f"\nCombined file: {self.config.output_dir}/combined.txt")

        print("=" * 60)


async def main_async() -> None:
    """Main async entry point."""
    args = parse_args()

    # Create config
    config = Config(
        input_source=args.input,
        output_dir=Path(args.output_dir),
        combine=args.combine,
        language=args.language,
        model_size=args.model_size,
        device=args.device,
        sleep_delay=args.sleep,
        verbose=args.verbose,
    )

    # Create transcriber
    transcriber = Transcriber(config)

    # Parse input type
    try:
        input_type, identifier, _ = parse_input(args.input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Process based on input type
    if input_type == "video":
        await transcriber.process_single_video(identifier)

    elif input_type == "playlist":
        playlist_url = f"https://www.youtube.com/playlist?list={identifier}"
        await transcriber.process_playlist(playlist_url)

    elif input_type == "file":
        await transcriber.process_file(identifier)

    # Create combined file if requested
    if config.combine:
        transcriber.create_combined_file()

    # Print summary
    transcriber.print_summary()

    # Exit with appropriate code
    if transcriber.failed:
        sys.exit(1)
    sys.exit(0)


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
