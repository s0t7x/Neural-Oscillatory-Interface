import numpy as np
import sounddevice as sd
import threading
import time
import random
import curses # Use curses for terminal input
import math # For ceiling function
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from rich.console import Console
from rich.measure import Measurement # To get component height

# Rich console setup needs to happen early
console = Console()

# === Parameters ===
sample_rate = 48000
base_freq = 400.0
beat_freq = 10.0
running = True

# Use smaller buffers for display to reduce memory/CPU
waveform_display_samples = 2048 # Keep ~40ms of samples for display
last_waveform_r = np.zeros(waveform_display_samples)
last_waveform_l = np.zeros(waveform_display_samples)

# Lock for frequency updates
freq_lock = threading.Lock()

# Presets: Label → (base_freq, beat_freq)
presets = {
    "Delta Sleep (1 Hz)": (100, 1),
    "Theta Meditate (4.5 Hz)": (300, 4.5), # Shortened label
    "Alpha Relax (8 Hz)": (400, 8),       # Shortened label
    "Beta Focus (16 Hz)": (500, 16),
    "Gamma Cognition (32 Hz)": (600, 32),
    "Schumann (7.83 Hz)": (250, 7.83),     # Shortened label
    "Deep Trance (2.5 Hz)": (220, 2.5),
    "Lucid Dream (14 Hz)": (440, 14),      # Shortened label
    "Memory (10 Hz)": (420, 10),          # Shortened label
}

preset_keys = list(presets.keys())

BLOCKS = "  ▂▃▄▅▆▇█"

def waveform_to_blocks(samples, width=60):
    """
    Convert waveform samples to block characters.
    Optimized for varying widths and handles empty data.
    """
    # Ensure width is at least 1 to avoid errors
    width = max(1, int(width))

    if samples is None or len(samples) == 0:
        return " " * width

    # Ensure we have a copy to avoid modifying the original buffer
    display_samples = samples.copy()

    if len(display_samples) < width:
        # If fewer samples than width, interpolate or pad
        # Simple padding:
        padded_samples = np.pad(display_samples, (0, width - len(display_samples)), 'constant', constant_values=np.mean(display_samples) if len(display_samples)>0 else 0)
        display_samples = padded_samples
    else:
        # Downsample using linspace to get representative points
        indices = np.linspace(0, len(display_samples) - 1, width, dtype=int)
        display_samples = display_samples[indices]

    # Normalize samples between 0 and 1
    min_val = np.min(display_samples)
    max_val = np.max(display_samples)
    range_val = max_val - min_val
    if range_val < 1e-6: # Avoid division by zero if flat or single point
        # Handle flat line: either all min or all max blocks, let's choose middle
        norm = np.full(len(display_samples), 0.5)
    else:
        norm = (display_samples - min_val) / range_val

    # Clamp normalized values just in case of float issues
    norm = np.clip(norm, 0.0, 1.0)

    num_blocks = len(BLOCKS)
    blocks = ''.join(BLOCKS[int(val * (num_blocks - 1))] for val in norm)
    return blocks

def table_freqs():
    global base_freq, beat_freq
    table = Table(title="Active Frequencies", title_style="bold magenta", border_style="magenta", box=None, show_header=False, padding=(0,1))
    table.add_column("Channel", justify="right", style="dim")
    table.add_column("Frequency (Hz)", justify="left")
    # Read frequencies safely
    with freq_lock:
        bf_local = base_freq
        beatf_local = beat_freq
    f1 = bf_local - beatf_local / 2
    f2 = bf_local + beatf_local / 2
    table.add_row("Left :", f"{f1:.2f}")
    table.add_row("Right:", f"{f2:.2f}")
    # Removed decorative +/- row
    return table

def table_presets():
    table = Table(title="Presets [1–9]", title_style="bold green", border_style="green", box=None, padding=(0,1))
    table.add_column("Key", justify="center", style="dim")
    table.add_column("Label", justify="left", no_wrap=True) # Prevent wrapping long labels
    table.add_column("Freqs", justify="center") # Combined freqs
    for i, (label, (bf, beat)) in enumerate(presets.items(), 1):
        table.add_row(f"[{i}]", label, f"{bf:.0f}/{beat:.1f}") # Compact freq display
    return table

def render_ui():
    """Creates the Rich Layout, adapts to console size."""
    global last_waveform_l, last_waveform_r

    layout = Layout()

    # Define main areas: Header, Body (splits later), Waveform, Footer
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1), # Takes remaining space
        Layout(name="waveforms", size=3), # Fixed size for waveform display
        Layout(name="footer", size=3)
    )

    # --- Header ---
    layout["header"].update(
        Panel(
            "[bold cyan]Cognitive Induction Terminal\n[white]Aural Pattern Induction", # Shortened subtitle
            title="Neural Oscillatory Interface",
            border_style="cyan",
            padding=(0,1)
        )
    )

    # --- Body: Decide stacking based on width ---
    presets_table = table_presets()
    freqs_table = table_freqs()

    layout["body"].split_column(
        Layout(presets_table, name="presets_col"),
        Layout(freqs_table, name="freqs_col")
    )

    # --- Waveforms ---
    # Calculate available width per waveform, accounting for panel borders/padding
    waveform_panel_padding = 2 # 1 left + 1 right from Panel
    column_padding = 1 # Space between columns if using Columns
    # Width available inside the panel for the two columns:
    inner_width = console.width - waveform_panel_padding
    # Width per waveform column:
    wf_width = (inner_width - column_padding) // 2
    wf_width = max(10, wf_width) # Ensure a minimum sensible width

    waveform_l_copy = last_waveform_l # No need to copy if audio thread updates carefully
    waveform_r_copy = last_waveform_r
    layout["waveforms"].update(
        Panel(Columns([waveform_to_blocks(waveform_l_copy, width=wf_width),
                       waveform_to_blocks(waveform_r_copy, width=wf_width)],
                      expand=True, equal=True, align="center"),
              title="Waveforms (L/R)", border_style="blue", padding=(0,0))
    )


    # --- Footer ---
    layout["footer"].update(
        Panel(
            "[dim]↑/↓ Beat | [1–9] Preset | [Esc] Quit", # Shortened footer
            border_style="green",
            padding=(0,1)
        )
    )

    return layout

def audio_loop():
    global base_freq, beat_freq, last_waveform_l, last_waveform_r, running
    chunk_size = 512 # Smaller chunks might feel more responsive
    t_offset = 0.0

    try:
        with sd.OutputStream(channels=2, samplerate=sample_rate, dtype='float32',
                             clip_off=True, dither_off=True, blocksize=chunk_size) as stream:
            while running:
                with freq_lock:
                    f1 = base_freq - beat_freq / 2
                    f2 = base_freq + beat_freq / 2

                # Generate time vector for the chunk
                t = (np.arange(chunk_size) + t_offset) / sample_rate
                left = 0.5 * np.sin(2 * np.pi * f1 * t)
                right = 0.5 * np.sin(2 * np.pi * f2 * t)

                # Update waveform buffers *carefully*
                # Concatenate and slice to keep the buffer size constant
                new_l = np.concatenate((last_waveform_l, left))
                new_r = np.concatenate((last_waveform_r, right))
                # Keep only the last N samples directly in the buffer
                last_waveform_l[:] = new_l[-waveform_display_samples:]
                last_waveform_r[:] = new_r[-waveform_display_samples:]


                stereo = np.stack((left, right), axis=-1).astype(np.float32)

                # Add a small sleep *before* writing to prevent busy-waiting if stream blocks
                # Adjust this value if audio is choppy or CPU usage is too high
                # time.sleep(chunk_size / sample_rate * 0.5) # Sleep for fraction of chunk duration

                stream.write(stereo)

                # Update time offset for seamless phase
                t_offset += chunk_size

    except sd.PortAudioError as e:
        # Specific handling for PortAudio errors which are common
        console.print(f"[bold red]PortAudio Error:[/bold red] {e}")
        console.print("[yellow]Is another application using the audio device?")
        console.print("[yellow]Try selecting a different audio device if possible.")
        running = False
    except Exception as e:
        console.print(f"[bold red]Audio Thread Error:[/bold red] {e}", )
        import traceback
        console.print(traceback.format_exc()) # Print stack trace for debugging
        running = False # Stop the application if audio fails


def main(stdscr): # stdscr is the screen object from curses
    global base_freq, beat_freq, running, console

    # --- Curses Setup ---
    curses.curs_set(0) # Hide the cursor
    stdscr.nodelay(True) # Make getch non-blocking
    stdscr.keypad(True) # Enable detection of special keys (like arrows)
    # Color setup (optional, Rich handles colors well)
    # curses.start_color()
    # curses.use_default_colors()

    # Start audio thread
    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()

    # Initial render
    current_layout = render_ui()

    stdscr.clear()
    stdscr.refresh()

    # Use Rich Live with the console object
    with Live(current_layout, console=console, refresh_per_second=15, screen=True) as live:
        while running:
            # --- Handle Input with Curses ---
            key = -1 # Reset key
            try:
                key = stdscr.getch() # Get character code (-1 if no key pressed)
            except curses.error:
                # This can happen during resize, ignore it and let KEY_RESIZE handle it
                pass
            except KeyboardInterrupt: # Handle Ctrl+C gracefully
                 running = False
                 break

            needs_redraw = True
            if key != -1:
                # --- Process Key Press ---
                if key == curses.KEY_UP:
                    with freq_lock:
                        beat_freq = min(beat_freq + 0.5, 100)
                    needs_redraw = True
                elif key == curses.KEY_DOWN:
                    with freq_lock:
                        beat_freq = max(beat_freq - 0.5, 0.1)
                    needs_redraw = True
                elif key == 27: # ASCII code for Escape key
                    running = False
                    break # Exit loop immediately
                elif ord('1') <= key <= ord('9'): # Check for number keys 1-9
                    idx = key - ord('1') # Calculate index (0-8)
                    if idx < len(preset_keys):
                        label = preset_keys[idx]
                        with freq_lock:
                            base_freq, beat_freq = presets[label]
                        needs_redraw = True

                # --- Handle Resize ---
                elif key == curses.KEY_RESIZE:
                    # Let Rich's console know the size might have changed
                    # console = Console() # Re-create console might be needed if size isn't updating
                    # Re-render the layout based on new dimensions
                    # stdscr.clear() # Clear curses buffer might help
                    # stdscr.refresh() # Refresh curses screen
                    stdscr.clear()
                    console = Console()
                    current_layout = render_ui() # Generate new layout based on new console size
                    live.update(current_layout, refresh=True) # Tell Live to use the new layout and redraw
                    # No need to set needs_redraw = True here, live.update handles it

            # --- Update Display ---
            # Only update if frequencies changed or periodically for waveforms
            # Live's refresh_per_second handles periodic updates
            if needs_redraw:
                 # We only need to update the tables if frequencies changed
                 # Re-rendering the whole layout also works and handles structure changes
                 current_layout = render_ui()
                 live.update(current_layout)
            elif not audio_thread.is_alive() and running:
                # If audio thread died unexpectedly, stop the main loop
                console.print("[bold red]Audio thread stopped unexpectedly. Exiting.")
                running = False
                break

            # Short sleep to prevent 100% CPU usage in the input loop
            # Live's refresh_per_second already introduces some delay
            time.sleep(1.0 / 30.0) # ~30 FPS check loop

    # --- Cleanup ---
    running = False # Ensure audio thread stops
    if audio_thread.is_alive():
        audio_thread.join(timeout=0.5) # Wait briefly for audio thread


if __name__ == "__main__":
    # Wrap the main function with curses.wrapper
    # It initializes curses, runs main, and cleans up curses afterwards
    try:
        curses.wrapper(main)
        print("\nExited cleanly.")
    except curses.error as e:
        print(f"\nTerminal error: {e}")
        print("Your terminal might not support curses features needed,")
        print("or it was resized too abruptly during execution.")
    except Exception as e:
        print(f"\nAn unexpected error occurred:")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure audio devices are released if sd hangs
        sd.stop()
        print("Sound stopped.")
        # Ensure running flag is false
        running = False