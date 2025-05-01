import numpy as np
import sounddevice as sd
import threading
import time
import random
from pynput import keyboard
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from rich.console import Console

console = Console()

# === Parameters ===
sample_rate = 48000
base_freq = 400.0
beat_freq = 10.0
running = True

last_waveform_r = []
last_waveform_l = []

# Lock for frequency updates
freq_lock = threading.Lock()

# Presets: Label → (base_freq, beat_freq)
presets = {
    "Delta Sleep (1 Hz)": (100, 1),
    "Theta Meditation (4.5 Hz)": (300, 4.5),
    "Alpha Relaxation (8 Hz)": (400, 8),
    "Beta Focus (16 Hz)": (500, 16),
    "Gamma Cognition (32 Hz)": (600, 32),
    "Schumann Resonance (7.83 Hz)": (250, 7.83),
    "Deep Trance (2.5 Hz)": (220, 2.5),
    "Lucid Dreaming (14 Hz)": (440, 14),
    "Memory Recall (10 Hz)": (420, 10),
}

preset_keys = list(presets.keys())

BLOCKS = " ▁▂▃▄▅▆▇█"

def waveform_to_blocks(samples, width=60):
    """
    Convert waveform samples to block characters.
    """
    if len(samples) < width:
        samples = np.pad(samples, (0, width - len(samples)), 'constant')
    else:
        samples = samples[:width]

    # Normalize samples between 0 and 1
    norm = (samples - samples.min()) / max((samples.max() - samples.min()), 1e-6)
    blocks = ''.join(BLOCKS[int(val * (len(BLOCKS) - 1))] for val in norm)
    return blocks

def table_freqs():
    global base_freq, beat_freq
    table = Table(title="Active Frequencies", title_style="bold magenta", border_style="magenta")
    table.add_column("Channel", justify="center")
    table.add_column("Frequency (Hz)", justify="center")
    f1 = base_freq - beat_freq / 2
    f2 = base_freq + beat_freq / 2
    table.add_row("Left", f"{f1:.2f}")
    table.add_row("Right", f"{f2:.2f}")
    table.add_row("+/-", f"0.0{random.randint(2,6):.0f}")
    return table

def table_presets():
    table = Table(title="Presets [1–9]", title_style="bold green", border_style="green")
    table.add_column("Key", justify="center")
    table.add_column("Label", justify="left")
    table.add_column("Base / Beat", justify="center")
    for i, (label, (bf, beat)) in enumerate(presets.items(), 1):
        table.add_row(str(i), label, f"{bf:.1f} / {beat:.2f}")
    return table

def render_ui():
    global base_freq, beat_freq, last_waveform_l, last_waveform_r
    layout = Layout()
    layout.split(
        Layout(name="header", size=5),
        Layout(name="body"),
        Layout(name="waveforms"),
        Layout(name="footer", size=3)
    )

    layout["header"].update(
        Panel(
            "[bold cyan]Cognitive Induction Terminal\n[white]Real-Time Aural Pattern Induction for Cognitive State Modulation",
            title="Neural Oscillatory Interface",
            border_style="cyan",
        )
    )

    layout["body"].update(
        Columns([table_presets(), table_freqs()], expand=True, align='center')
    )

    layout["waveforms"].update(
        Columns([waveform_to_blocks(last_waveform_l, width=80), waveform_to_blocks(last_waveform_r, width=80)], expand=True)
    )

    layout["footer"].update(
        Panel(
            "[dim]Use ↑ / ↓ to adjust beat frequency • [1–9] to select preset • [Esc] to quit",
            border_style="green"
        )
    )

    return layout

def audio_loop():
    def generate():
        global base_freq, beat_freq, last_waveform_l, last_waveform_r
        while running:
            with freq_lock:
                f1 = base_freq - beat_freq / 2
                f2 = base_freq + beat_freq / 2
            t = np.linspace(0, 1, sample_rate, False)
            left = 0.5 * np.sin(2 * np.pi * f1 * t)
            last_waveform_l = left
            right = 0.5 * np.sin(2 * np.pi * f2 * t)
            last_waveform_r = right
            stereo = np.stack((left, right), axis=-1).astype(np.float32)
            
            yield stereo

    try:
        with sd.OutputStream(channels=2, samplerate=sample_rate, dtype='float32', clip_off=True, dither_off=True, latency=0.1, blocksize=2) as stream:
            for block in generate():
                stream.write(block)
                if not running:
                    break
    except Exception as e:
        console.print(f"[red]Audio error: {e}")

def keyboard_listener():
    global base_freq, beat_freq, running
    def on_press(key):
        global base_freq, beat_freq, running

        try:
            if key == keyboard.Key.esc:
                running = False
                return False
            elif key == keyboard.Key.up:
                with freq_lock:
                    beat_freq = min(beat_freq + 0.5, 100)
            elif key == keyboard.Key.down:
                with freq_lock:
                    beat_freq = max(beat_freq - 0.5, 0.1)
            elif hasattr(key, 'char') and key.char in "123456789":
                idx = int(key.char) - 1
                if idx < len(preset_keys):
                    label = preset_keys[idx]
                    with freq_lock:
                        base_freq, beat_freq = presets[label]
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

def main():
    global running
    threading.Thread(target=audio_loop, daemon=True).start()
    keyboard_listener()

    with Live(render_ui()) as live:
        while running:
            time.sleep(0.5)
            live.update(render_ui())

if __name__ == "__main__":
    main()
