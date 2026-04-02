
# Local-Agentic-Orchestrator - Training a quad virtual dog 
High-performance local LLM orchestration on 16GB Unified Memory with multimodal screen interaction training a quad virtual dog.
<video src="https://github.com/user-attachments/assets/e00d5813-8c15-430a-8c35-2b692ad33077" controls="controls" width="100%"></video>

# High-performance local LLM orchestration on 16GB Unified Memory with multimodal screen interaction some early bugs being addressed.
<video src="https://github.com/user-attachments/assets/61ee7283-83a2-4165-bc5b-51fdca72ce01" controls="controls" width="100%"></video>

# Android project, random balls that are affected when you shake the physical phone, they have gravity, and collisions :)
<video src="https://github.com/user-attachments/assets/6b97af8b-12cf-4b94-8fc5-0c52bd16b305" controls="controls" width="100%"></video>

# 2019 MacBook running Ubuntu Linux Native.  Project: control trackbar and animate.
<video src="https://github.com/user-attachments/assets/ed3673a4-a5d3-46e7-9385-63119a126987" controls="controls" width="100%"></video>

# test
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bubble Simulation</title>
    <style>
        /* Add your CSS here */
        body { margin: 0; overflow: hidden; background: #000; }
        canvas { display: block; }
        .controls { position: absolute; top: 10px; left: 10px; z-index: 10; }
        button { padding: 10px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="controls">
        <button id="addBtn">+ Add 5 Bubbles</button>
        <button id="popAllBtn">Pop All</button>
    </div>
    <canvas id="bubbleCanvas"></canvas>

    <script>
        // 1. SETUP: Canvas, Context, and State
        const canvas = document.getElementById('bubbleCanvas');
        const ctx = canvas.getContext('2d');
        let audioCtx = null;
        let bubbles = [];

        // 2. PHYSICS: Elastic collisions & Wall bounce
        // Use mass = radius^2 for your collision calculations
        function resolveCollision(b1, b2) {
            // Your collision response logic goes here
        }

        // 3. VISUALS: Iridescence & Shimmer
        // Use ctx.createRadialGradient for the film look
        function drawBubble(bubble) {
            // Your rendering logic (gradients, highlights, shimmer)
        }

        // 4. SOUND: AudioContext synthesis
        function playPopSound() {
            if (!audioCtx) return;
            // Your filtered noise burst logic
        }

        // 5. INTERACTION & LOOP
        window.addEventListener('click', () => {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        });

        // 6. POP PHASE: 2-minute timer
        setTimeout(() => {
            // Logic to start the random popping phase
        }, 120000);

        // ... (Rest of your simulation loop and event listeners)
    </script>
</body>
</html>


