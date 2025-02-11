# Peter AI 🚀

Hey, friend! Welcome to **Peter AI (me)**, your go-to personal assistant that’s all about making your life easier. No more repeating yourself. Peter’s here to remember everything you tell him. We’re leveraging some awesome tech like **Pinecone** and **OpenAI**, and I can’t wait to roll out even cooler features in the future!

## What Can Peter Do? 🤖 

- **Long-Term Memory**: Peter’s like your best buddy who remembers all the important stuff, so you can focus on what really matters.
- **Smart Search**: Need to find something? With Pinecone, you’ll be zooming through your info in no time.
- **Natural Language Processing**: Thanks to OpenAI, Peter gets you no more robotic responses!
- **Future Features**: We’ve got big plans ahead! Think local AI models and databases. Stay tuned!

## Getting Started 🛠️ 

### Prerequisites 
Make sure you’ve got **Node.js** ready to go on your machine. 

### Installation 
Let’s get Peter up and running with a few simple steps:

1. **Install Dependencies**:  
   Open your terminal and run:  
   ```sh
   npm install  
   ```

2. **Set Up Environment Variables**:  
   You’ll need some keys to get rolling. Just do this in your terminal:  
   ```sh
   echo "OPENAI_API_KEY=sk-proj-1234567890" >> .env  
   echo "PINECONE_API_KEY=pcsk_fq" >> .env  
   ```

3. **Initialize Pinecone Index**:  
   If you haven’t set up your index yet, go for it:  
   ```sh
   NODE_OPTIONS="--loader ts-node/esm" ts-node setup-pinecone.ts  
   ```

4. **Fire Up Peter AI**:  
   Time to bring Peter to life:  
   ```sh
   NODE_OPTIONS="--loader ts-node/esm" ts-node main.ts  

   ```
## Current Limitations 🔍 

Right now, Peter only supports console interactions. Stay tuned for future updates that will broaden the ways you can connect with your trusty assistant!

## What’s Next? 🚧 

Here’s what’s bubbling up for Peter:  
- Local vectorized database integration  
- Support for custom AI models  
- Better memory organization  
- Integration with tools like email and calendars  

## Wanna Contribute? 🙌 

We’d love to have you on board! Feel free to fork the repo, send pull requests, or just drop any ideas or issues you think of. Your input is what makes Peter better!

---

Written with love by me, **Peter AI** your friendly assistant! 🎉