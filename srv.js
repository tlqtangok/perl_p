const http = require('http');

const port = process.argv[2] || 3000;

function getTimestamp() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hour = String(now.getHours()).padStart(2, '0');
  const min = String(now.getMinutes()).padStart(2, '0');
  return `${year}${month}${day}_${hour}${min}`;
}

function log(msg) {
  console.log(`[${getTimestamp()}] ${msg}`);
}

const server = http.createServer((req, res) => {
  const clientAddr = req.socket.remoteAddress;
  const clientPort = req.socket.remotePort;
  log(`SERVER: Connection from ${clientAddr}:${clientPort}`);
  log(`SERVER: ${req.method} ${req.url}`);
  
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  
  const body = `[${getTimestamp()}] hello world\n`;
  res.end(body);
  log(`SERVER: Sent ${body.length} bytes`);
});

server.listen(port, () => {
  log(`Server running at http://localhost:${port}/`);
});
