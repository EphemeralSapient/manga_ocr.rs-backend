#!/usr/bin/env python3
"""
Manga text processing workflow client
Supports single file, folder processing, and interactive/flag-based CLI
"""

import argparse
import json
import base64
import time
import sys
from pathlib import Path
from typing import Optional, Dict, List
import requests

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box
from rich.prompt import Confirm
import questionary
from questionary import Style, Choice

# Initialize rich console
console = Console()

# Custom questionary style
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
])


def browse_files(start_path: Path = None, allow_files: bool = True) -> Optional[Path]:
    """Visual file/folder browser"""
    if start_path is None:
        start_path = Path.cwd()

    current_path = start_path.resolve()
    supported_formats = {'.webp', '.png', '.jpg', '.jpeg'}

    while True:
        choices = []

        # Add parent directory option if not at root
        if current_path.parent != current_path:
            choices.append(Choice(title="📁 .. (Parent Directory)", value=".."))

        # Add current directory option (for selecting folder)
        choices.append(Choice(title=f"✓ Select this folder: {current_path.name}", value="."))

        choices.append(Choice(title="─" * 50, disabled=True))

        try:
            # List directories first
            dirs = sorted([d for d in current_path.iterdir() if d.is_dir() and not d.name.startswith('.')],
                         key=lambda x: x.name.lower())
            for d in dirs:
                choices.append(Choice(title=f"📁 {d.name}/", value=d))

            # Then list files if allowed
            if allow_files:
                files = sorted([f for f in current_path.iterdir()
                              if f.is_file() and f.suffix.lower() in supported_formats],
                              key=lambda x: x.name.lower())
                for f in files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    choices.append(Choice(
                        title=f"📄 {f.name} [{size_mb:.1f}MB]",
                        value=f
                    ))
        except PermissionError:
            console.print("[red]Permission denied for this directory[/red]")
            return None

        if len(choices) == 3:  # Only parent, current, and separator
            choices.append(Choice(title="(Empty directory)", disabled=True))

        choices.append(Choice(title="─" * 50, disabled=True))
        choices.append(Choice(title="⎋ Cancel", value=None))

        selection = questionary.select(
            f"Browse: {current_path}",
            choices=choices,
            style=custom_style,
            use_shortcuts=True,
            use_arrow_keys=True
        ).ask()

        if selection is None:
            return None
        elif selection == "..":
            current_path = current_path.parent
        elif selection == ".":
            return current_path
        elif isinstance(selection, Path):
            if selection.is_dir():
                current_path = selection
            else:
                return selection
        else:
            return None


class MangaClient:
    """Client for manga text processing workflow"""

    SUPPORTED_FORMATS = {'.webp', '.png', '.jpg', '.jpeg'}

    def __init__(self, server_url: str = "http://localhost:1429"):
        self.server_url = server_url
        self.process_url = f"{server_url}/process"
        self.health_url = f"{server_url}/health"
        self.console = console

    def wait_for_server(self, timeout: int = 60) -> bool:
        """Wait for server to be ready"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Waiting for server...", total=None)
            start = time.time()

            while time.time() - start < timeout:
                try:
                    if requests.get(self.health_url, timeout=2).status_code == 200:
                        progress.update(task, description="[green]✓ Server ready")
                        time.sleep(0.3)  # Brief pause to show success
                        return True
                except requests.RequestException:
                    time.sleep(2)

            progress.update(task, description="[red]✗ Server timeout")
            return False

    def process_image(self, image_path: Path, config: Dict) -> Optional[Dict]:
        """Process single image"""
        if not image_path.exists():
            self.console.print(f"[red]✗ Not found: {image_path}[/red]")
            return None

        files = {'images': (image_path.name, open(image_path, 'rb'), f'image/{image_path.suffix[1:]}')}
        data = {'config': json.dumps(config)}

        try:
            response = requests.post(self.process_url, files=files, data=data, timeout=300)
            if response.status_code == 200:
                return response.json()
            else:
                self.console.print(f"[red]✗ Server error: {response.status_code}[/red]")
                return None
        except Exception as e:
            self.console.print(f"[red]✗ Error: {e}[/red]")
            return None
        finally:
            files['images'][1].close()

    def process_folder(self, folder_path: Path, config: Dict, output_dir: Path) -> List[Dict]:
        """Process all images in folder"""
        images = [f for f in folder_path.iterdir() if f.suffix.lower() in self.SUPPORTED_FORMATS]

        if not images:
            self.console.print(f"[red]✗ No images found in {folder_path}[/red]")
            return []

        self.console.print(Panel(f"[cyan]Processing {len(images)} images from {folder_path}[/cyan]",
                                 title="[bold]Batch Processing[/bold]", border_style="cyan"))
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            overall_task = progress.add_task("[cyan]Processing images...", total=len(images))

            for i, img in enumerate(images, 1):
                progress.update(overall_task, description=f"[cyan]Processing {img.name}...")
                start = time.time()
                result = self.process_image(img, config)

                if result:
                    elapsed = time.time() - start
                    self.console.print(f"  [green]✓[/green] {img.name} [dim]({elapsed:.2f}s)[/dim]")
                    self.save_result(result, output_dir, img.stem)
                    results.append(result)
                else:
                    self.console.print(f"  [red]✗[/red] {img.name} [dim](failed)[/dim]")

                progress.advance(overall_task)

        return results

    def save_result(self, result: Dict, output_dir: Path, prefix: str = "processed"):
        """Save processed images and analytics"""
        output_dir.mkdir(exist_ok=True)

        # Save analytics
        analytics_file = output_dir / f"{prefix}_analytics.json"
        with open(analytics_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Save images
        saved_count = 0
        for page in result.get('results', []):
            if page.get('success') and page.get('data_url'):
                data_url = page['data_url']
                if data_url.startswith('data:image/png;base64,'):
                    base64_data = data_url.replace('data:image/png;base64,', '')
                    image_data = base64.b64decode(base64_data)

                    filename = page.get('filename', f"page_{page.get('index', 0)}")
                    output_file = output_dir / f"{prefix}_{Path(filename).stem}.png"

                    with open(output_file, 'wb') as f:
                        f.write(image_data)
                    saved_count += 1

        if saved_count > 0:
            self.console.print(f"    [dim]💾 Saved {saved_count} image(s) to {output_dir}[/dim]")

    def print_analytics(self, result: Dict):
        """Print analytics summary with rich table"""
        analytics = result.get('analytics', {})

        # Create regions table
        regions_table = Table(title="📊 Processing Analytics", box=box.ROUNDED, border_style="cyan")
        regions_table.add_column("Metric", style="cyan", no_wrap=True)
        regions_table.add_column("Value", style="magenta", justify="right")

        regions_table.add_row("Total Regions", str(analytics.get('total_regions', 0)))
        regions_table.add_row("Simple Background", str(analytics.get('simple_bg_count', 0)))
        regions_table.add_row("Complex Background", str(analytics.get('complex_bg_count', 0)))

        # API calls
        api_simple = analytics.get('api_calls_simple', 0)
        api_complex = analytics.get('api_calls_complex', 0)
        api_banana = analytics.get('api_calls_banana', 0)
        regions_table.add_row("API Calls", f"{api_simple} + {api_complex} + {api_banana} = {api_simple + api_complex + api_banana}")

        # Cache stats
        cache_hits = analytics.get('cache_hits', 0)
        cache_misses = analytics.get('cache_misses', 0)
        total_cache = cache_hits + cache_misses
        cache_rate = f"{(cache_hits/total_cache*100):.1f}%" if total_cache > 0 else "N/A"
        regions_table.add_row("Cache", f"{cache_hits}/{total_cache} ({cache_rate})")

        # Timing table
        timing_table = Table(title="⏱️  Phase Timings", box=box.ROUNDED, border_style="green")
        timing_table.add_column("Phase", style="green")
        timing_table.add_column("Time", style="yellow", justify="right")

        timing_table.add_row("Phase 1 (Detection)", f"{analytics.get('phase1_time_ms', 0):.0f} ms")
        timing_table.add_row("Phase 2 (Segmentation)", f"{analytics.get('phase2_time_ms', 0):.0f} ms")
        timing_table.add_row("Phase 3 (Translation)", f"{analytics.get('phase3_time_ms', 0):.0f} ms")
        timing_table.add_row("Phase 4 (Rendering)", f"{analytics.get('phase4_time_ms', 0):.0f} ms")

        total_time = sum([
            analytics.get('phase1_time_ms', 0),
            analytics.get('phase2_time_ms', 0),
            analytics.get('phase3_time_ms', 0),
            analytics.get('phase4_time_ms', 0)
        ])
        timing_table.add_row("[bold]Total[/bold]", f"[bold]{total_time:.0f} ms ({total_time/1000:.2f}s)[/bold]")

        self.console.print()
        self.console.print(regions_table)
        self.console.print()
        self.console.print(timing_table)


def interactive_mode(client: MangaClient):
    """Interactive CLI mode with beautiful prompts"""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Manga Text Processing Workflow[/bold cyan]\n[dim]Interactive Mode - Use arrow keys to navigate[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))

    # Browse for input file or folder
    console.print("\n[cyan]Step 1:[/cyan] Select input file or folder")
    input_path = browse_files()

    if not input_path:
        console.print("[yellow]Operation cancelled[/yellow]")
        return

    # Get output directory
    console.print("\n[cyan]Step 2:[/cyan] Select output directory")
    output_choice = questionary.select(
        "How do you want to specify the output directory?",
        choices=[
            Choice("📁 Use default (./output)", value="default"),
            Choice("🔍 Browse for folder", value="browse"),
            Choice("⌨️  Type custom path", value="custom")
        ],
        style=custom_style
    ).ask()

    if not output_choice:
        console.print("[yellow]Operation cancelled[/yellow]")
        return

    if output_choice == "default":
        output_dir = Path("output")
    elif output_choice == "browse":
        output_dir = browse_files(allow_files=False)
        if not output_dir:
            console.print("[yellow]Operation cancelled[/yellow]")
            return
    else:  # custom
        output_dir_str = questionary.text(
            "Enter output directory path:",
            default="output",
            style=custom_style
        ).ask()
        if not output_dir_str:
            console.print("[yellow]Operation cancelled[/yellow]")
            return
        output_dir = Path(output_dir_str)

    # Get font family with choices
    console.print("\n[cyan]Step 3:[/cyan] Select font family")
    font = questionary.select(
        "Choose font:",
        choices=[
            Choice("🔤 Arial (Default)", value="arial"),
            Choice("🎨 Comic Sans", value="comic-sans"),
            Choice("🈂️  Noto Sans Mono CJK", value="noto-sans-mono-cjk"),
            Choice("⌨️  Custom font...", value="custom")
        ],
        style=custom_style
    ).ask()

    if not font:
        console.print("[yellow]Operation cancelled[/yellow]")
        return

    if font == "custom":
        font = questionary.text(
            "Enter custom font name:",
            style=custom_style
        ).ask()
        if not font:
            console.print("[yellow]Operation cancelled[/yellow]")
            return

    config = {"font_family": font}

    # Wait for server
    if not client.wait_for_server():
        return

    # Process
    console.print()
    start_time = time.time()

    if input_path.is_file():
        with console.status("[cyan]Processing image...", spinner="dots"):
            result = client.process_image(input_path, config)

        if result:
            client.save_result(result, output_dir, input_path.stem)
            client.print_analytics(result)
            elapsed = time.time() - start_time
            console.print()
            console.print(Panel(f"[green]✓ Processing completed in {elapsed:.2f}s[/green]",
                              border_style="green"))
        else:
            console.print(Panel("[red]✗ Processing failed[/red]", border_style="red"))
    else:
        results = client.process_folder(input_path, config, output_dir)
        elapsed = time.time() - start_time

        if results:
            # Aggregate analytics
            total_regions = sum(r.get('analytics', {}).get('total_regions', 0) for r in results)
            total_api_calls = sum(
                r.get('analytics', {}).get('api_calls_simple', 0) +
                r.get('analytics', {}).get('api_calls_complex', 0) +
                r.get('analytics', {}).get('api_calls_banana', 0)
                for r in results
            )

            summary_table = Table(title="Batch Summary", box=box.ROUNDED, border_style="green")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta", justify="right")
            summary_table.add_row("Images Processed", str(len(results)))
            summary_table.add_row("Total Regions", str(total_regions))
            summary_table.add_row("Total API Calls", str(total_api_calls))
            summary_table.add_row("Total Time", f"{elapsed:.2f}s")
            summary_table.add_row("Avg Time/Image", f"{elapsed/len(results):.2f}s")

            console.print()
            console.print(summary_table)
            console.print()
            console.print(Panel(f"[green]✓ Batch processing completed successfully![/green]",
                              border_style="green"))
        else:
            console.print(Panel("[red]✗ No images processed[/red]", border_style="red"))


def main():
    parser = argparse.ArgumentParser(
        description="Manga text processing client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  %(prog)s

  # Process single file
  %(prog)s -i input.webp

  # Process folder
  %(prog)s -i images/ -o results/

  # With custom font
  %(prog)s -i input.webp -f "comic-sans" -o output/
        """
    )

    parser.add_argument('-i', '--input', type=Path, help='Input file or folder')
    parser.add_argument('-o', '--output', type=Path, default=Path('output'), help='Output directory (default: output)')
    parser.add_argument('-f', '--font', default='arial', help='Font family (default: arial)')
    parser.add_argument('-s', '--server', default='http://localhost:1429', help='Server URL (default: http://localhost:1429)')
    parser.add_argument('--no-wait', action='store_true', help='Skip server health check')

    args = parser.parse_args()

    # Initialize client
    client = MangaClient(args.server)

    # Interactive mode if no input provided
    if not args.input:
        interactive_mode(client)
        return

    # Validate input
    if not args.input.exists():
        console.print(f"[red]✗ Not found: {args.input}[/red]")
        sys.exit(1)

    # Wait for server
    if not args.no_wait and not client.wait_for_server():
        sys.exit(1)

    config = {"font_family": args.font}

    # Process
    console.print()
    console.print(Panel(f"[cyan]Processing {args.input}[/cyan]", border_style="cyan"))
    start_time = time.time()

    if args.input.is_file():
        with console.status("[cyan]Processing image...", spinner="dots"):
            result = client.process_image(args.input, config)

        if result:
            client.save_result(result, args.output, args.input.stem)
            client.print_analytics(result)
            elapsed = time.time() - start_time
            console.print()
            console.print(Panel(f"[green]✓ Completed in {elapsed:.2f}s[/green]",
                              border_style="green"))
        else:
            console.print(Panel("[red]✗ Processing failed[/red]", border_style="red"))
            sys.exit(1)
    else:
        results = client.process_folder(args.input, config, args.output)
        elapsed = time.time() - start_time

        if results:
            total_regions = sum(r.get('analytics', {}).get('total_regions', 0) for r in results)
            total_api_calls = sum(
                r.get('analytics', {}).get('api_calls_simple', 0) +
                r.get('analytics', {}).get('api_calls_complex', 0) +
                r.get('analytics', {}).get('api_calls_banana', 0)
                for r in results
            )

            summary_table = Table(title="Batch Summary", box=box.ROUNDED, border_style="green")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="magenta", justify="right")
            summary_table.add_row("Images Processed", str(len(results)))
            summary_table.add_row("Total Regions", str(total_regions))
            summary_table.add_row("Total API Calls", str(total_api_calls))
            summary_table.add_row("Total Time", f"{elapsed:.2f}s")
            summary_table.add_row("Avg Time/Image", f"{elapsed/len(results):.2f}s")

            console.print()
            console.print(summary_table)
            console.print()
            console.print(Panel(f"[green]✓ Batch processing completed successfully![/green]",
                              border_style="green"))
        else:
            console.print(Panel("[red]✗ No images processed[/red]", border_style="red"))
            sys.exit(1)


if __name__ == "__main__":
    main()
