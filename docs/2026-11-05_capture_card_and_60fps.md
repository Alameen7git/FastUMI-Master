# Capture card swaps and true 60 fps recording (2026-11-05)

Questions and answers about the changes made on 2026-11-05 to the FastUMI web UI and recorder
setup, why they were needed, and how the 60 fps problem was found and fixed.

**Short version**
- The camera used to work with only one specific Elgato card. Now it finds any Elgato capture
  card by itself, and restarts on its own when the card is swapped or re-plugged.
- Recordings with the new card were coming out at about 50 fps, not 60. Now they are recorded at
  a true 60 fps, as long as the card is on a USB 3 port.
- There is a new tool, `viz/check_fps.py`, that checks the real frame rate of any recording.

---

## Part 1: Swapping the capture card

### Q: What was the problem?
After replacing the Elgato capture card with another one, the camera never turned on. The web UI
showed the Camera light failing, and nothing came through, even though the card was plugged in
and working.

### Q: Why did a different card of the same model not work?
The camera node (`usb_cam`, started by `usb_cam-test.launch`) opened the device `/dev/elgato`.
That name is not created by Linux on its own: it comes from a udev rule on the NUC,
`/etc/udev/rules.d/99-elgato.rules`, which only matches USB product id `0fd9:00ae`. The
replacement card is the same model (Elgato 4K S) but reports product id `0fd9:00af`. So
`/dev/elgato` was never created, and `usb_cam` had no device to open.

Two cards of the same model can differ in product id (hardware revision) and always differ in
serial number, so anything tied to one card's ids breaks when the card is swapped.

### Q: What does the web UI do now?
When the camera starts, the web UI searches for the card itself, instead of relying on
`/dev/elgato`. It picks the video capture device of any Elgato card (USB vendor `0fd9`), whatever
its model or serial, and tells `usb_cam` which device to open (for example `/dev/video0`).

The Machines page shows which card is in use under the Camera light, and the log prints it, e.g.:

    [GoPro] capture card: Elgato 4K S (serial AC8KB5261E7WXH) USB 3 at /dev/video0, sending YUYV

### Q: What happens if I unplug or swap the card while the camera is on?
The web UI checks for the card every second:

| What you do | What happens |
|---|---|
| Unplug the card while the camera is on | The camera stops. |
| Plug a card in (the same one or another) | After about 2 seconds the camera starts again, on that card. |
| Swap straight from one card to another | The camera restarts on the new card. |
| Press Start with no card plugged in | The page says "Plug in the capture card: the camera starts by itself once it is plugged in", and it does. |
| Swap the card during a recording | No automatic restart, so a recording is never interrupted. The Record page's restart button uses the new card, and once the recording ends the camera switches to it by itself. |
| Swap the card after pressing Stop | Nothing starts. The camera only follows the card while you want it on. |

### Q: Does the old terminal workflow (`start_collection.sh`) also work with any card?
Not by itself. It still opens `/dev/elgato`, which only exists for the one card the udev rule
matches. To make `/dev/elgato` work for any Elgato card, run this once on the NUC (it asks for
the password):

    echo 'SUBSYSTEM=="video4linux", ATTRS{idVendor}=="0fd9", ATTR{index}=="0", SYMLINK+="elgato"' | sudo tee /etc/udev/rules.d/99-elgato.rules && sudo udevadm control --reload && sudo udevadm trigger

The web UI does not need this.

### Q: How was this tested?
- On the real card: the web UI found it, and `usb_cam` opened it and published 1920×1080 frames.
- Swapping: plugging, unplugging, swapping, starting with no card, recording, and Stop were
  simulated (the card could not be physically unplugged during testing). All 12 checks passed.

---

## Part 2: Recording at a true 60 fps

### Q: How did we find out the recordings were not 60 fps?
`video.mp4` always says 60 fps, because the recorder writes the file with a fixed frame rate, so
it cannot be trusted. What counts is `raw/episode_N/timestamps.csv`, which has the time each frame
actually arrived. The new `viz/check_fps.py` measures the real rate from those times.

Test recordings with the new card:

| Dataset | Real frame rate | Frames missing | Camera mode |
|---|---|---|---|
| `UMI_2026-09-29_T1_N2` (old card, for comparison) | 59.95–59.99 fps | 0% | MJPEG, USB 2 |
| `UMI_2026-11-05_T0_N1` | 50.5–50.8 fps | 15–16% | MJPEG, USB 3 |
| `UMI_2026-11-05_T0_N2` | 49.0–49.9 fps | 17–18% | MJPEG, USB 3 (before the UI was restarted with the fix) |
| `UMI_2026-11-05_T0_N3` | **59.95–59.96 fps** | **0.1%** | **YUYV, USB 3 (the fix)** |

The T265 was at 199.9 Hz in every recording, so only the video was affected.

### Q: Was the GoPro or the capture card the problem?
No. With nothing else running, the card itself was measured delivering a steady 60.00 fps with a
live GoPro picture, in both MJPEG and YUYV mode. The frames were lost on the NUC, after the card.

### Q: So what was losing the frames?
Decoding the JPEG frames. The camera node asked the card for MJPEG (compressed JPEG frames).
On USB 3 this card sends very large, high-quality JPEGs: about **890 KB per frame**, roughly
430 Mbit/s. The old card was on USB 2, which limited how big each frame could be.

`usb_cam` decodes each JPEG on a single CPU core. With these large frames:

- decoding alone takes about 14 ms per frame (600 frames in 8.5 s);
- `usb_cam` also has to colour-convert and publish each frame;
- at 60 fps there is only 16.7 ms per frame, so it does not fit.

With only ROS and the camera running, `usb_cam` reached about 54 fps. With the recorder and T265
also running, recordings came out at 49–51 fps.

### Q: What is the fix?
On USB 3, the web UI now asks the card for **uncompressed video (YUYV)** instead of MJPEG. YUYV
needs no JPEG decoding at all, only a cheap colour conversion.

| Mode | Camera alone | With the recorder's full workload |
|---|---|---|
| MJPEG (before) | 54.0 fps | 53.5 fps |
| YUYV (now) | 59.9 fps | **59.95 fps** (900 of 900 frames in 15 s) |

In YUYV mode `usb_cam` uses about half of one CPU core.

The "recorder's full workload" test copied exactly what `data_collection.py` does with each frame
(same subscriber settings, colour conversion, frozen-feed check, and mp4 writer). A real
recording afterwards (`UMI_2026-11-05_T0_N3`) confirmed it: 59.95 fps, and in 600 consecutive
frames no frame was a repeat of the one before, so the GoPro itself is outputting 60 fps.

The picture is identical in both modes, colours included.

### Q: Why not always use YUYV?
Uncompressed 1080p60 is about 2 Gbit/s (1920 × 1080 × 2 bytes × 60), and USB 2 carries at most
480 Mbit/s. So the web UI picks the mode from the USB speed it detects:

- **USB 3:** YUYV (the fix).
- **USB 2:** MJPEG, the only mode that fits. The old card reached 60 fps this way, but the
  current card's large JPEGs do not, so **keep the card on a USB 3 port**.

`FASTUMI_PIXEL_FORMAT=yuyv` or `=mjpeg` forces one mode if ever needed.

### Q: Which port should the card be plugged into?
A **USB 3 port** on the NUC: usually blue inside, or marked **SS** (SuperSpeed). Use the card's
own cable, with no hub in between. To check:

- In the web UI log, when the camera starts, the line should end with
  **`USB 3 at /dev/video0, sending YUYV`**. If it says `USB 2` and `sending MJPEG`, move the card
  to another port or try another cable.
- Or in a terminal: `lsusb -t`. The card's `uvcvideo` lines should show `5000M` (USB 3), not
  `480M` (USB 2).

### Q: I updated the code but my recordings are still at 50 fps. Why?
The web UI keeps running the code it started with. After any update, close the FastUMI Control
terminal window and click the desktop icon again. This is what happened with
`UMI_2026-11-05_T0_N2`: the UI had been started before the fix, so the camera was still in MJPEG.

### Q: How do I check a recording's frame rate?

    python3 viz/check_fps.py                          # newest dataset
    python3 viz/check_fps.py UMI_2026-11-05_T0_N3     # a specific dataset
    python3 viz/check_fps.py UMI_2026-11-05_T0_N3 --episode 2

Example output:

    episode   1:  59.95 fps   19.9s   1194 frames  median 16.65 ms  worst   22.8 ms  missing   1 (0.1%)  T265 199.9 Hz  OK

| Column | Good value | Meaning |
|---|---|---|
| fps | 59.5–60 | Real frame rate, from the arrival times. |
| median | about 16.7 ms | The usual time between frames. |
| worst | under ~35 ms | The longest gap between two frames. |
| missing | 0–1 | Frames a steady 60 fps would have had in that time, minus the frames that arrived. |
| T265 | about 200 Hz | Tracking rate. |
| OK / LOW | OK | LOW means under ~58 fps or more than about 1 frame in 500 missing. |

It also warns if `video.mp4` and `timestamps.csv` have different frame counts.

"Missing" is counted from the total, not from long gaps. A late frame followed by an early one
is normal timing jitter and does not lose a frame.

### Q: What about the datasets recorded at 50 fps?
They were not mislabelled. Sync measures the real camera rate and keeps every second frame, so
`UMI_2026-11-05_T0_N1` was saved as 25 fps synced data, not 30. Still, `UMI_2026-11-05_T0_N1` and
`UMI_2026-11-05_T0_N2` were test runs with 15–18% of frames missing; delete them (Delete tab)
rather than train on them.

### Q: The GoPro's "NO SD CARD" and battery icons are in the picture. Is that related?
No, that is a GoPro setting. The icons come from the GoPro's HDMI output in *Monitor* mode. On
the GoPro, set Preferences → Input/Output → HDMI Output to **Live**.

---

## Files changed

| File | Change |
|---|---|
| `webui/app.py` | Finds the capture card; follows card swaps; picks YUYV or MJPEG by USB speed. |
| `webui/settings.py` | Capture card settings: vendors, `FASTUMI_CAPTURE_DEVICE`, pixel format per USB speed, `FASTUMI_PIXEL_FORMAT`. |
| `webui/templates/nodes.html` | Shows the capture card under the Camera light, and "plug in the capture card" when waiting. |
| `usb_cam-test.launch` | Takes `video_device:=` and `pixel_format:=` (defaults unchanged: `/dev/elgato`, `mjpeg`). |
| `viz/check_fps.py` | New: measures the real frame rate of recordings. |
| `README.md` | New section 1.6: plug the capture card into a USB 3 port, and how to check. |
| `webui/README.md` | Card swapping and the USB 3 / YUYV note. |
