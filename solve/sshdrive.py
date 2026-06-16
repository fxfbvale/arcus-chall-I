"""PTY driver for the Arcus SSH TUI. Usage: python3 sshdrive.py [KEYS...]
e.g. python3 sshdrive.py ENTER     (open the challenge detail screen)
Renders the alt-screen TUI with pyte and prints the final screen + raw bytes."""
import os, sys, pty, select, time, pyte, fcntl, termios, struct

KEYMAP = {"ENTER": b"\r", "TAB": b"\t", "ESC": b"\x1b", "SPACE": b" ",
          "UP": b"\x1b[A", "DOWN": b"\x1b[B", "RIGHT": b"\x1b[C", "LEFT": b"\x1b[D",
          "CTRLC": b"\x03", "Q": b"q", "BS": b"\x7f"}
if len(sys.argv) >= 3 and sys.argv[1] == "--submitfile":
    flag = open(sys.argv[2], "r", encoding="utf-8").read()
    SCRIPT = [(3.0, b""), (1.5, b"\r"), (1.5, flag.encode("utf-8")), (1.0, b""),
              (1.5, b"\r"), (3.0, None)]
elif len(sys.argv) >= 3 and sys.argv[1] in ("--submit", "--type"):
    # open challenge, type the flag string; --submit also presses ENTER to submit
    flag = sys.argv[2]
    SCRIPT = [(3.0, b""), (1.5, b"\r"), (1.5, flag.encode("utf-8")), (1.5, None)] \
        if sys.argv[1] == "--type" else \
        [(3.0, b""), (1.5, b"\r"), (1.5, flag.encode("utf-8")), (1.0, b""),
         (1.5, b"\r"), (3.0, None)]
else:
    SCRIPT = [(3.0, b"")]
    for a in sys.argv[1:]:
        SCRIPT.append((1.2, KEYMAP.get(a.upper(), a.encode())))
    SCRIPT.append((2.5, None))

COLS, ROWS = 120, 40
screen = pyte.Screen(COLS, ROWS); stream = pyte.ByteStream(screen)
pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = "xterm-256color"
    os.execvp("ssh", ["ssh", "-tt", "-o", "StrictHostKeyChecking=accept-new",
                      "-o", "ConnectTimeout=15", "augustalabs.ai"])
    os._exit(1)
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
raw = bytearray()
def pump(dur):
    end = time.time() + dur
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.2)
        if fd in r:
            try: data = os.read(fd, 65536)
            except OSError: return False
            if not data: return False
            raw.extend(data); stream.feed(data)
    return True
alive = True
for delay, keys in SCRIPT:
    if not alive: break
    alive = pump(delay)
    if keys is None: break
    if keys and alive:
        try: os.write(fd, keys)
        except OSError: alive = False
try: os.write(fd, b"\x03"); os.write(fd, b"q")
except OSError: pass
time.sleep(0.3)
try: os.close(fd)
except OSError: pass
print("===RENDERED SCREEN===")
for line in screen.display:
    print(line.rstrip())
print("===END===")
open("/tmp/ssh_raw.bin", "wb").write(raw)
sys.stderr.write(f"[raw bytes: {len(raw)}]\n")
