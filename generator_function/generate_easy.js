const { createCanvas } = require("canvas");
const fs = require("fs");
const path = require("path");

// Configuration
const canvasWidth = 300;
const canvasHeight = 300;
const fontSize = 50;
const textColor = "#000000"; // Black text
const backgroundColor = "#FFFFFF"; // White background
const font = "Verdana"; // Only one font
const outputDir = path.join(__dirname, "/images/easy");

// Ensure output directory exists
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

function capitalizeFirstLetter(word) {
    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
}

function drawFilledText(ctx, text) {
    // Fill the background with plain white
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    ctx.font = `${fontSize}px ${font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = textColor; // Black fill color
    ctx.strokeStyle = textColor; // Black outline color
    ctx.lineWidth = 2; // Stroke thickness

    const x = canvasWidth / 2;
    const y = canvasHeight / 2;
    ctx.fillText(text, x, y);
    ctx.strokeText(text, x, y);
}

function saveImage(canvas, filePath) {
    const buffer = canvas.toBuffer("image/png");
    fs.writeFileSync(filePath, buffer);
}

function processWordlist(filename) {
    const words = fs.readFileSync(filename, "utf-8").split(/\r?\n/).filter(word => word.trim());
    let wordCount = 0;

    words.forEach((word, i) => {
        const formattedWord = capitalizeFirstLetter(word);

        const canvas = createCanvas(canvasWidth, canvasHeight);
        const ctx = canvas.getContext("2d");

        drawFilledText(ctx, formattedWord);

        const filename = `${formattedWord}_${i + 1}.png`;
        const filePath = path.join(outputDir, filename);
        saveImage(canvas, filePath);

        wordCount++;
        console.log(`Saved ${wordCount}/100`);
    });

    console.log(`\nTotal Images Generated: ${wordCount}`);
}

processWordlist("custom_word_list_100.txt");
