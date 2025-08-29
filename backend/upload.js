import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// Fix for __dirname in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Set up multer storage to save files in backend/uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadPath = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadPath)) {
      fs.mkdirSync(uploadPath);
    }
    cb(null, uploadPath);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

// POST /api/upload
router.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  // Save the prompt to a .txt file in the uploads folder
  const prompt = req.body.prompt || '';
  if (prompt) {
    const promptFileName = req.file.filename.replace(/\.[^/.]+$/, '') + '-prompt.txt';
    const promptFilePath = path.join(path.dirname(req.file.path), promptFileName);
    fs.writeFileSync(promptFilePath, prompt, 'utf8');
  }

  res.json({ message: 'File and prompt uploaded successfully', filePath: req.file.path });
});

export default router;
