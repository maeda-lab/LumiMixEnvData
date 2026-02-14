const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;
let timerState = 'stopped'; // 初始状态为 'stopped'


// 设置静态文件目录
app.use(express.static(path.join(__dirname, 'public')));

// 路由
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 跨域设置，允许不同浏览器访问
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

// 获取计时器状态
app.get('/status', (req, res) => {
    res.json({ status: timerState });
});

// 启动计时器
app.post('/start', (req, res) => {
    timerState = 'running';
    res.json({ message: 'Timer started' });
});

// 停止计时器
app.post('/stop', (req, res) => {
    timerState = 'stopped';
    res.json({ message: 'Timer stopped' });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
