function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

function escapeHTML(str) {
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#039;");
}


async function sendUserInput() {
    // Function that sends user input to the server and awaits a response
    let userInput = document.getElementById('user-input').value.trim(); // Trimmed user input
    let chatBox = document.getElementById('chat-box'); // Chat box element

    if (!userInput) {
        alert("Please enter a message!");
        return;
    }

    // Escape input to prevent XSS
    userInput = escapeHTML(userInput);

    // Post user message
    let userMessage = document.createElement("div");
    userMessage.className = "message user-message";
    userMessage.innerText = userInput;
    chatBox.appendChild(userMessage);

    console.log("User Input:", userInput);
    console.log(analyzeUserInputUrl);

    try {
        const response = await fetch(analyzeUserInputUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({ message: userInput })
        });
        const data = await response.json();

        // ✅ Create a div for the bot response
        let botMessage = document.createElement("div");
        botMessage.className = "message bot-message";
        botMessage.innerText = data.message;  //
        chatBox.appendChild(botMessage); //
        console.log("Server Response:", data);
    } catch (error) {
        console.error("Error sending message:", error);
    }
}