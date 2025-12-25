# 保存为 zhuaqu_simple.py
import random, requests
from pathlib import Path

OUT_DIR = Path("one_random_video"); OUT_DIR.mkdir(exist_ok=True)

VIDEO_POOL = [
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
]

url = random.choice(VIDEO_POOL)
vid_path = OUT_DIR / "video.mp4"

print(f"🎬 正在下载视频：{url}")
with requests.get(url, stream=True, timeout=30) as r:
    r.raise_for_status()
    with open(vid_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
print("✅ 视频下载完成：", vid_path)

# 保存说明文字
with open(OUT_DIR / "caption.txt", "w", encoding="utf-8") as f:
    f.write(f"Sample video clip: {url}\n")

print("✅ 文本已保存：", OUT_DIR / "caption.txt")
print("🎉 全部完成！文件保存在：", OUT_DIR.resolve())
