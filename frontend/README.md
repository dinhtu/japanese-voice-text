# Pronunciation Practice — Frontend

Giao diện luyện phát âm tiếng Nhật cho [API đánh giá phát âm](../README.md#pronunciation-evaluation-api).

Người dùng đọc to câu tiếng Nhật hiển thị sẵn, bấm micro để ghi âm, và nhận điểm 0–100.

## Stack

Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS v4 · lucide-react

## Chạy

Cần backend chạy trước ở `http://localhost:8000`:

```bash
# terminal 1 — API (ở thư mục gốc của repo)
uv run uvicorn app.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Mở <http://localhost:3000>.

> Ghi âm cần `localhost` hoặc HTTPS — trình duyệt chặn micro trên HTTP domain thường.

### Cấu hình

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL của API |

Sao chép `.env.example` thành `.env.local` để đổi. Nếu đổi port của frontend, nhớ thêm origin đó vào `CORS_ORIGINS` của backend.

## Cấu trúc

```
src/
├── app/
│   ├── layout.tsx           # root layout + metadata
│   ├── globals.css          # design tokens (tông xanh dương), animation
│   └── page.tsx
├── components/
│   ├── ui/                  # primitives: button, card
│   └── pages/pronunciation/
│       ├── pronunciation-practice.tsx  # container: state + gọi API
│       ├── target-text-card.tsx        # câu mục tiêu + chọn câu khác
│       ├── recorder-panel.tsx          # nút micro, đồng hồ, nghe lại
│       ├── result-card.tsx             # điểm, diff kana, danh sách lỗi
│       └── score-ring.tsx              # vòng tròn điểm số
├── hooks/use-audio-recorder.ts  # micro → WAV, mức âm lượng trực tiếp
├── lib/
│   ├── api/pronunciation.ts     # POST /api/pronunciation/evaluate
│   ├── wav.ts                   # encode WAV 16kHz mono
│   └── utils.ts                 # cn(), formatDuration()
├── constants/practice-texts.ts  # câu luyện tập (hard-code)
├── types/pronunciation.ts       # mirror schema của API
└── config/index.ts
```

## Đổi câu luyện tập

Sửa [`src/constants/practice-texts.ts`](src/constants/practice-texts.ts). Câu có thể viết bằng kanji, kana hoặc trộn — backend tự chuẩn hoá về cách đọc hiragana trước khi so sánh:

```ts
{
  id: "aisatsu",
  text: "おはようございます",
  reading: "ohayou gozaimasu",
  meaning: "Chào buổi sáng.",
}
```

Phần tử đầu tiên là câu mặc định khi mở trang.

## Ghi âm → WAV

`MediaRecorder` chỉ xuất webm/ogg, trong khi API chỉ nhận `.wav`. Nên [`lib/wav.ts`](src/lib/wav.ts) decode bản ghi bằng Web Audio API, downmix về mono, resample xuống 16 kHz (đúng sample rate của model) rồi tự ghi RIFF header — không cần thư viện ngoài.

Ghi âm tự dừng ở 30 giây để file không vượt giới hạn upload của API.

## Kiểm tra

```bash
npm run typecheck
npm run lint
npm run build
```
