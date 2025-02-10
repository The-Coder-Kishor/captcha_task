const { createCanvas } = require("canvas");
const fs = require("fs");
const path = require("path");

// Configuration
const canvasWidth = 300;
const canvasHeight = 300;
const fontSize = 50;
const colors = ["#800080", "#0000FF", "#008000", "#FFFF00", "#FF0000"]; // VBGYR
const fonts = ["Comic Sans MS", "Courier New", "Verdana"]; // Three distinct fonts
const outputDir = path.join(__dirname, "/images/hard/random");

// Ensure output directory exists
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

function generateCapitalizations(word) {
    const variations = new Set();
    const n = word.length;
    for (let i = 0; i < (1 << n); i++) {
        let variation = "";
        for (let j = 0; j < n; j++) {
            variation += (i & (1 << j)) ? word[j].toUpperCase() : word[j].toLowerCase();
        }
        variations.add(variation);
    }
    return Array.from(variations);
}

function generateRGBNoise(ctx) {
    const imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    for (let i = 0; i < imageData.data.length; i += 4) {
        imageData.data[i] = 200 + Math.random() * 55;     // Red
        imageData.data[i + 1] = 200 + Math.random() * 55; // Green
        imageData.data[i + 2] = 200 + Math.random() * 55; // Blue
        imageData.data[i + 3] = 255;                      // Alpha
    }
    ctx.putImageData(imageData, 0, 0);
}

function drawHollowText(ctx, text, font, outlineColor) {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    generateRGBNoise(ctx);
    
    ctx.font = `${fontSize}px ${font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.strokeStyle = outlineColor;

    for (let i = 0; i < text.length; i++) {
        const offsetX = Math.random() * 10 - 5;
        const offsetY = Math.random() * 10 - 5;
        ctx.lineWidth = Math.random() * 1.5 + 0.5;  // Reduced lineWidth
        ctx.strokeText(
            text[i],
            canvasWidth / 2 + (i - text.length / 2) * fontSize * 0.6 + offsetX,
            canvasHeight / 2 + offsetY
            // canvasWidth / 2 + (i - text.length / 2) * fontSize * 0.6,
            // canvasHeight / 2
        );
    }
}

function saveImage(canvas, filePath) {
    const buffer = canvas.toBuffer("image/png");
    fs.writeFileSync(filePath, buffer);
}

function processWordlist(filename) {
    const words = fs.readFileSync(filename, "utf-8").split(/\r?\n/).filter(word => word.trim());
    let wordCount = 0;
    
    words.forEach(word => {
        const capitalizations = generateCapitalizations(word);
        var i = 0;
        capitalizations.forEach(capWord => {
            fonts.forEach(font => {
                colors.forEach((color, colorIndex) => {
                    i++;
                    const canvas = createCanvas(canvasWidth, canvasHeight);
                    const ctx = canvas.getContext("2d");
                    
                    drawHollowText(ctx, capWord, font, color);
                    
                    const filename = `${word}_${i}.png`;
                    const filePath = path.join(outputDir, filename);
                    saveImage(canvas, filePath);
                    
                });
            });
        });
        wordCount++;
        console.log(`Saved ${wordCount}/100`);
    });

    console.log(`\nTotal Images Generated: ${wordCount}`);
}

processWordlist("custom_word_list_100.txt");
// processWordlist("hello.txt");
