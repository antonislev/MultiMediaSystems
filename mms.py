import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import heapq

# Load video path from Downloads
video_path = os.path.expanduser("~/Downloads/3076_THE_SOCIAL_NETWORK_01.33.57.154-01.34.01.880.avi")

def extract_frames(video_path, output_dir="frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frames = []

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        cv2.imwrite(f"{output_dir}/frame_{idx:03d}.png", gray)
        idx += 1
    cap.release()
    return np.array(frames)

frames = extract_frames(video_path)

def frame_differences(frames):
    diffs = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        diffs.append(diff)
    return np.array(diffs)

diffs = frame_differences(frames)

for i in range(5): 
    plt.imshow(diffs[i], cmap='gray')
    plt.title(f"Διαφορά Πλαισίου {i}")
    plt.show()

plt.imshow(frames[50], cmap='gray')  # 50ο πλαίσιο
plt.title("Frame 50")
plt.show()

flat_diffs = diffs.flatten()
print(flat_diffs[:20])

freq = Counter(flat_diffs)
print(freq)

def compute_huffman_dict(data):
    freq = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return dict(sorted(heap[0][1:], key=lambda p: (len(p[-1]), p)))

huff_dict = compute_huffman_dict(flat_diffs)

print("Λεξικό Huffman για τις πιο συχνές τιμές:")
for symbol, code in list(huff_dict.items())[:20]:
    print(f"Σύμβολο: {symbol}, Κώδικας: {code}")

huff_size = sum(len(huff_dict[symbol]) * freq[symbol] for symbol in freq)
original_size = flat_diffs.size * 8
compression_ratio = original_size / huff_size

print(f"Αρχικό μέγεθος (σε bits): {original_size}")
print(f"Μέγεθος μετά τη συμπίεση (σε bits): {huff_size}")
print(f"Λόγος συμπίεσης: {compression_ratio}")

def motion_compensation(ref_frame, target_frame, block_size=8, search_range=4):
    height, width = ref_frame.shape
    predicted_frame = np.zeros_like(ref_frame)
    motion_vectors = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            best_match = (0, 0)
            min_error = float('inf')

            target_block = target_frame[y:y+block_size, x:x+block_size]

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_y = y + dy
                    ref_x = x + dx

                    if ref_y < 0 or ref_y + block_size > height or ref_x < 0 or ref_x + block_size > width:
                        continue

                    ref_block = ref_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                    error = np.sum(np.abs(target_block - ref_block))

                    if error < min_error:
                        min_error = error
                        best_match = (dy, dx)

            ref_y = y + best_match[0]
            ref_x = x + best_match[1]
            predicted_frame[y:y+block_size, x:x+block_size] = ref_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
            motion_vectors.append(((y, x), best_match))

    return predicted_frame, motion_vectors

predicted_frames = []
motion_vectors_all = []

frames_short = extract_frames(video_path)[:5]
frames = frames_short

for i in range(1, len(frames)):
    predicted, vectors = motion_compensation(frames[i-1], frames[i], block_size=8, search_range=4)
    predicted_frames.append(predicted)
    motion_vectors_all.append(vectors)

motion_diffs = [cv2.absdiff(frames[i+1], predicted_frames[i]) for i in range(len(predicted_frames))]
motion_diffs = np.array(motion_diffs)
flat_motion_diffs = motion_diffs.flatten()

freq = Counter(flat_motion_diffs)
huff_dict = compute_huffman_dict(flat_motion_diffs)
huff_size = sum(len(huff_dict[symbol]) * freq[symbol] for symbol in freq)
original_size = flat_motion_diffs.size * 8
compression_ratio = original_size / huff_size

print(f"[Motion Comp] Αρχικό μέγεθος (σε bits): {original_size}")
print(f"[Motion Comp] Συμπιεσμένο μέγεθος (σε bits): {huff_size}")
print(f"[Motion Comp] Λόγος συμπίεσης: {compression_ratio}")
