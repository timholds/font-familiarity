// font-validator.js - Script to check which Google Font URLs work and which need weight parameters
const fs = require('fs');
const https = require('https');

// Read the available_fonts.txt file
const fontList = fs.readFileSync('./static/available_fonts.txt', 'utf8')
    .split('\n')
    .filter(line => line.trim())
    .map(line => line.trim());

console.log(`Loaded ${fontList.length} fonts from available_fonts.txt`);

// Weights to try for fonts that fail with the default URL
const weightsToTry = ['300', '400', '500', '700', '900'];

// Test if a URL works by making a HEAD request
function testUrl(url) {
    return new Promise((resolve) => {
        const req = https.request(url, { method: 'HEAD' }, (res) => {
            resolve({
                url,
                status: res.statusCode,
                success: res.statusCode === 200
            });
        });
        
        req.on('error', () => {
            resolve({
                url,
                status: 'error',
                success: false
            });
        });
        
        req.end();
    });
}

// Test a font with different URL patterns
async function testFont(fontName) {
    // Test the standard URL
    const standardUrl = `https://fonts.googleapis.com/css2?family=${fontName.replace(/\s+/g, '+')}`;
    const standardResult = await testUrl(standardUrl);
    
    if (standardResult.success) {
        return { 
            font: fontName, 
            needsWeight: false, 
            workingUrl: standardUrl 
        };
    }
    
    // If standard URL fails, try with weights
    for (const weight of weightsToTry) {
        const weightUrl = `https://fonts.googleapis.com/css2?family=${fontName.replace(/\s+/g, '+')}:wght@${weight}`;
        const weightResult = await testUrl(weightUrl);
        
        if (weightResult.success) {
            return { 
                font: fontName, 
                needsWeight: true, 
                weight, 
                workingUrl: weightUrl 
            };
        }
    }
    
    // If all tests fail
    return { 
        font: fontName, 
        needsWeight: false, 
        error: true,
        message: 'All URL patterns failed' 
    };
}

// Main function to test all fonts
async function checkAllFonts() {
    const results = {
        standardFonts: [],
        weightSpecificFonts: [],
        errorFonts: []
    };
    
    // Add small delay between requests to avoid rate limiting
    async function processWithDelay(fonts) {
        const processed = [];
        
        for (let i = 0; i < fonts.length; i++) {
            const font = fonts[i];
            console.log(`Testing ${i+1}/${fonts.length}: ${font}`);
            
            const result = await testFont(font);
            processed.push(result);
            
            if (result.needsWeight) {
                results.weightSpecificFonts.push(result);
            } else if (result.error) {
                results.errorFonts.push(result);
            } else {
                results.standardFonts.push(result);
            }
            
            // Small delay to avoid overwhelming the server
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        return processed;
    }
    
    await processWithDelay(fontList);
    
    // Generate output for fontWeightMap
    console.log('\n=== FONT WEIGHT MAP ===');
    console.log('const fontWeightMap = {');
    results.weightSpecificFonts.forEach(font => {
        console.log(`  '${font.font.toLowerCase().replace(/\s+/g, '')}': '${font.weight}',`);
    });
    console.log('};');
    
    // Log summary
    console.log('\n=== SUMMARY ===');
    console.log(`Total fonts tested: ${fontList.length}`);
    console.log(`Standard fonts (no weight needed): ${results.standardFonts.length}`);
    console.log(`Fonts needing specific weights: ${results.weightSpecificFonts.length}`);
    console.log(`Fonts with errors: ${results.errorFonts.length}`);
    
    // Save detailed results to a file
    fs.writeFileSync('font-validation-results.json', JSON.stringify(results, null, 2));
    console.log('\nDetailed results saved to font-validation-results.json');
}

// Run the script
checkAllFonts().catch(console.error);