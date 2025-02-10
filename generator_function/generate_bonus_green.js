const { createCanvas } = require("canvas");
const fs = require("fs");
const path = require("path");

// Configuration
const canvasWidth = 300;
const canvasHeight = 150;
const fontSize = 50;
const colors = ["#000000"]; // Black outline
const fonts = ["Comic Sans MS", "Courier New", "Verdana"];
const outputDir = path.join(__dirname, "/images/bonus/green");

// Ensure output directory exists
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

function generateCapitalizations(word) {
    const variations = new Set();
    const n = word.length;
    const maxVariations = Math.min(16, 1 << n);

    while (variations.size < maxVariations) {
        let variation = "";
        for (let j = 0; j < n; j++) {
            variation += Math.random() < 0.5 ? word[j].toUpperCase() : word[j].toLowerCase();
        }
        variations.add(variation);
    }

    return Array.from(variations);
}

function generateGreenNoise(ctx) {
    const imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    for (let i = 0; i < imageData.data.length; i += 4) {
        imageData.data[i + 1] = 100; // Green channel (dominant)
        imageData.data[i + 3] = 90;                 // Lower opacity
    }
    ctx.putImageData(imageData, 0, 0);
}

function drawHollowText(ctx, text, font, outlineColor) {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Set bright green background
    ctx.fillStyle = "#00FF00"; 
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Add green noise
    generateGreenNoise(ctx);

    ctx.font = `${fontSize}px ${font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.strokeStyle = outlineColor;
    ctx.lineWidth = 4; // Thicker outline

    const startX = canvasWidth / 2;
    const startY = canvasHeight / 2;

    // Slight distortions per character
    for (let i = 0; i < text.length; i++) {
        const offsetX = (i - text.length / 2) * fontSize * 0.6 + (Math.random() - 0.5) * 5;
        const offsetY = (Math.random() - 0.5) * 5;
        ctx.strokeText(text[i], startX + offsetX, startY + offsetY);
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
                colors.forEach((color) => {
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