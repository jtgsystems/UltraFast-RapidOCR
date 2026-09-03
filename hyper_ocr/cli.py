import argparse
import sys
import os
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from .engine import HyperOCREngine
from .screen import ScreenCapturer
from .ai_corrector import decrypt_handwriting_with_vision

console = Console()

def copy_to_clipboard(text: str):
    """Attempt copying text to OS clipboard."""
    import subprocess
    if sys.platform == "linux":
        for cmd in [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True)
                return True
            except Exception:
                continue
    elif sys.platform == "darwin":
        try:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True)
            return True
        except Exception:
            pass
    return False

def main():
    parser = argparse.ArgumentParser(
        prog="hyper-ocr",
        description="JTG Systems - HyperOCR: Ultra-Fast Real-Time OCR & AI Handwriting Decryptor (SOTA 2026)"
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # Scan command
    scan_p = subparsers.add_parser("scan", help="Scan an image file or screenshot")
    scan_p.add_argument("file", help="Path to image file")
    scan_p.add_argument("--ai", action="store_true", help="Enable Vision AI handwriting decryption for messy scribbles")
    scan_p.add_argument("--copy", action="store_true", help="Copy extracted text to clipboard")
    
    # Screen capture command
    snip_p = subparsers.add_parser("snip", help="Capture full screen or region and extract text")
    snip_p.add_argument("--monitor", type=int, default=1, help="Monitor index (default: 1)")
    snip_p.add_argument("--copy", action="store_true", default=True, help="Copy extracted text to clipboard")
    
    # Bench command
    bench_p = subparsers.add_parser("bench", help="Run hardware speed benchmark")
    
    args = parser.parse_args()
    
    if args.command == "scan":
        if not os.path.exists(args.file):
            console.print(f"[red]Error: File '{args.file}' not found![/red]")
            sys.exit(1)
            
        if args.ai:
            console.print(f"🧠 [bold purple]AI Vision Handwriting Mode:[/bold purple] Decrypting {args.file}...")
            ai_text = decrypt_handwriting_with_vision(args.file)
            if ai_text:
                console.print(Panel(ai_text, title="[bold green]✓ Decrypted Handwriting Transcription[/bold green]", border_style="green"))
                if args.copy:
                    copy_to_clipboard(ai_text)
                    console.print("[bold green]✓ Copied transcription to clipboard![/bold green]")
                return
            else:
                console.print("[yellow]Vision AI offline, falling back to Fast Optical OCR...[/yellow]")
                
        img = cv2.imread(args.file)
        if img is None:
            console.print(f"[red]Error: Could not decode image '{args.file}'![/red]")
            sys.exit(1)
            
        console.print(f"🔍 [bold cyan]Scanning:[/bold cyan] {args.file} ({img.shape[1]}x{img.shape[0]})")
        engine = HyperOCREngine()
        results, latency = engine.recognize(img)
        
        table = Table(title=f"Extracted Text ({len(results)} lines in {latency:.2f} ms)", border_style="cyan")
        table.add_column("Confidence", style="bold green", width=12)
        table.add_column("Text Content", style="white")
        
        full_text = []
        for r in results:
            table.add_row(f"{r['confidence'] * 100:.1f}%", r["text"])
            full_text.append(r["text"])
            
        console.print(table)
        
        if args.copy and full_text:
            combined = "\n".join(full_text)
            if copy_to_clipboard(combined):
                console.print("[bold green]✓ Copied all text to clipboard![/bold green]")
                
    elif args.command == "snip":
        console.print("📸 [bold cyan]Capturing screen buffer...[/bold cyan]")
        capturer = ScreenCapturer()
        img = capturer.capture_monitor(args.monitor)
        
        engine = HyperOCREngine()
        results, latency = engine.recognize(img)
        
        table = Table(title=f"Screen Text Extractor ({len(results)} lines in {latency:.2f} ms)", border_style="green")
        table.add_column("Confidence", style="bold green", width=12)
        table.add_column("Text Content", style="white")
        
        full_text = []
        for r in results:
            table.add_row(f"{r['confidence'] * 100:.1f}%", r["text"])
            full_text.append(r["text"])
            
        console.print(table)
        
        if full_text:
            combined = "\n".join(full_text)
            if copy_to_clipboard(combined):
                console.print("[bold green]✓ Text copied to clipboard![/bold green]")
                
    elif args.command == "bench":
        console.print(Panel("[bold yellow]⚡ HyperOCR Hardware Speed Benchmark[/bold yellow]\nTesting CUDA GPU & SIMD Pipeline...", border_style="yellow"))
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_img[:] = (245, 245, 245)
        cv2.putText(test_img, "JTG Systems Enterprise Workstations", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 2)
        cv2.putText(test_img, "HyperOCR Ultra-Fast Real-Time OCR Engine", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 50, 180), 2)
        cv2.putText(test_img, "Hardware Accelerated Optical Text Detection", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 20, 50), 2)
        
        engine = HyperOCREngine()
        # Warmup
        engine.recognize(test_img)
        
        runs = 15
        import time
        t0 = time.perf_counter()
        for _ in range(runs):
            res, _ = engine.recognize(test_img, use_cache=False)
        avg_ms = (time.perf_counter() - t0) / runs * 1000
        
        # Test Cache
        t0 = time.perf_counter()
        for _ in range(100):
            res_c, _ = engine.recognize(test_img, use_cache=True)
        cache_ms = (time.perf_counter() - t0) / 100 * 1000
        
        console.print(f"[bold green]✓ Steady-State Latency:[/bold green] {avg_ms:.2f} ms ({1000/avg_ms:.1f} FPS)")
        console.print(f"[bold cyan]✓ Frame Cache Latency:[/bold cyan]  {cache_ms:.3f} ms ({1000/cache_ms:.1f} FPS)")
        console.print(f"[bold white]Hardware Profile:[/bold white] GPU Available: {engine.gpu_available}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
