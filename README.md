# Nifty लाइव ऑप्शन चेन डैशबोर्ड

Upstox की लाइव Nifty ऑप्शन चेन, AI सपोर्ट/रेजिस्टेंस, सप्लाई-डिमांड ज़ोन, liquidity और बैकटेस्ट.
सिर्फ़ विश्लेषण के लिए है, ट्रेड की सलाह नहीं.

## क्या-क्या है
- **ऑप्शन चेन**: Calls लाल / Puts हरे, Volume और OI के % बार, Break Out / Break Down सिग्नल, पिछले ~5 मिनट का OI बदलाव,
  हर स्ट्राइक का liquidity स्कोर (OI, Volume, bid-ask spread).
- **AI विश्लेषण**: सपोर्ट, रेजिस्टेंस, रिवर्सल प्राइस, Max Pain, PCR, OI वॉल और अलर्ट.
- **चार्ट**: Nifty कैंडल (1/5/15 मिनट), Supply/Demand ज़ोन (स्कोर के साथ), EQH/EQL/PDH/PDL liquidity और sweep.
- **दो मोड** (साइडबार): *फोकस* (ज़रूरी चीज़ें) और *फुल* (सब कुछ).
- **AI व्याख्या (वैकल्पिक)**: Claude ज़ोन और OI की व्याख्या करता है (Anthropic API Key चाहिए).
- **Zone Backtest पेज**: सप्लाई/डिमांड ज़ोन को Upstox की पिछले 28 दिन की 1-मिनट हिस्ट्री पर परखता है.

## लोकल चलाना
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # फिर अपने token भरें
streamlit run app.py
```
Token secrets में न डालें तो साइडबार में चिपका सकते हैं.

## GitHub पर नई repository
```bash
git init
git add .
git commit -m "Nifty option chain dashboard"
git branch -M main
git remote add origin https://github.com/<आपका-username>/<repo-नाम>.git
git push -u origin main
```
`.gitignore` में `.streamlit/secrets.toml` पहले से है, फिर भी push से पहले `git status` में देख लीजिए कि कोई token वाली फ़ाइल नहीं जा रही.

## Streamlit Community Cloud पर चलाना (वैकल्पिक)
1. share.streamlit.io पर GitHub से लॉगिन करके repo चुनें, Main file: `app.py`.
2. Advanced settings -> Secrets में `.streamlit/secrets.toml.example` जैसे अपने token चिपकाएँ.
3. Repo को **Private** रखिए.

## ज़रूरी बातें
- **Upstox token रोज़ expire होता है.** रोज़ नया बनाकर secrets (या साइडबार) में डालना होगा.
- 5 मिनट के OI बदलाव की history सर्वर की मेमोरी में रहती है. ऐप सो जाए या restart हो तो मिट जाती है.
- सारे नियम (सिग्नल, ज़ोन, स्कोर) इस प्रोजेक्ट के अपने हैं, किसी इंडिकेटर की कॉपी नहीं. पहले बैकटेस्ट और कुछ दिन असली चाल से मिलाकर परखिए.
- Nifty इंडेक्स में volume नहीं आता, इसलिए ज़ोन में volume का इस्तेमाल नहीं है.
