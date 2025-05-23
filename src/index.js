// index.js

import express from 'express';
import cors from 'cors'; 

const app = express();
const PORT = process.env.PORT || 3000;

// CORS 설정 (모든 요청 허용)
app.use(cors());

// JSON 파싱 미들웨어
app.use(express.json());

// 기본 라우트
app.get('/', (req, res) => {
  res.send('Hello from PuppySense!');
});

// 서버 시작
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
