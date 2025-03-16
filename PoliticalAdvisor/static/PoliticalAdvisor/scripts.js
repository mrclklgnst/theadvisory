function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

document.getElementById('userInput').addEventListener('keypress', function (event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default form submission
        document.querySelector('button[onclick="sendUserInput()"]').click(); // Trigger the button click
    }
});
// Set the language selector to the language stored in the cookie
document.getElementById('languageSelector').value = getLanguageFromCookie();


function changeLanguage(language) {
    // Save the selected language to a cookie or session storage
    document.cookie = "language=" + language + "; path=/";
    location.reload();  // Reload the page to apply the new language
}
// Function to get the language from the cookie
function getLanguageFromCookie() {
    const name = "language=";
    const decodedCookie = decodeURIComponent(document.cookie);
    const ca = decodedCookie.split(';');
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i].trim();
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "en";  // Default to English if no language cookie exists
}

function scrollToBottom() {
    const messageWindow = document.getElementById('messageWindow');
    const lastChild = messageWindow.lastElementChild;
    if (lastChild) {
        lastChild.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function escapeHTML(str) {
    return str.replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function displayBotMessage(data) {
    let botMessage = document.createElement("div");
    botMessage.className = "message bot-message";

    // Set the inner text depending on format of the data
    if (typeof data === "string") {
        botMessage.innerText = data;
    } else if (typeof data === "object") {

        let introText = document.createElement("p");
        introText.innerText = languageContext["tableTitle"];
        botMessage.appendChild(introText);

        botMessage.appendChild(createPartyList(data));

    }
    // show bot message
    document.getElementById('messageWindow').appendChild(botMessage);
    scrollToBottom();

}
let currentCarouselIndex = 0;

function createPartyList(data) {
    let container = document.createElement("div");
    container.className = "party-list"; // Apply custom styles

    for (let party in data) {
        let partyContainer = document.createElement("div");
        partyContainer.className = "party-entry p-3 border-bottom"; // Bootstrap classes for spacing and divider

        // Party Name & Relevance
        let header = document.createElement("div");
        header.className = "d-flex justify-content-between align-items-center"; // Flexbox for spacing

        // Only add the Citations button if there are citations
        let hasCitations = data[party].citations && data[party].citations.length > 0;
        let headerContent = `
            <strong>${party.toUpperCase()}</strong> 
            <span class="ml-2">Relev.: ${data[party].agreement}%</span>
        `;
        header.innerHTML = headerContent;

        if (hasCitations) {
            let button = document.createElement("button");
            button.innerText = "Citations";
            button.className = "btn btn-dark btn-sm";
            button.onclick = function () {
                openCitationModal(party, data[party].citations);
            };
            header.appendChild(button);
        }

        // Explanation
        let explanation = document.createElement("p");
        explanation.className = "mt-2"; // Bootstrap styling
        explanation.innerText = data[party].explanation;

        // Append elements
        partyContainer.appendChild(header);
        partyContainer.appendChild(explanation);
        container.appendChild(partyContainer);
    }

    return container;
}

function botInitialMessage() {
    let botMessage = document.createElement("div");
    botMessage.className = "message bot-message";

    let introText = lang_context["bot_init_message"];
    botMessage.innerText = introText;

    // show bot message
    document.getElementById('messageWindow').appendChild(botMessage);
    scrollToBottom();
}

function openCitationModal(party, citations) {
    document.getElementById("modal-title").innerText = `Citations for ${party.toUpperCase()}`;
    let carouselContainer = document.getElementById("citationCarousel");
    carouselContainer.innerHTML = "";  // Clear existing citations

    citationItems = [];  // Reset citations
    currentCitationIndex = 0;

    // Check if citations exist and have length
    if (!citations || citations.length === 0) {
        let noCitationsDiv = document.createElement("div");
        noCitationsDiv.className = "no-citations";
        noCitationsDiv.innerHTML = `<p>No citations available for ${party.toUpperCase()}</p>`;
        carouselContainer.appendChild(noCitationsDiv);
        document.getElementById("citationModal").style.display = "flex";
        return;
    }

    // Create citation elements
    for (let i in citations) {
        let citationDiv = document.createElement("div");
        citationDiv.className = "carousel-item";

        // Ensure all citation properties exist
        const source = citations[i].source || "Unknown";
        const location = citations[i].location || "Unknown";
        const content = citations[i].content || "No content available";

        citationDiv.innerHTML = `
            <div class="citation">
                <strong>${source.toUpperCase()}</strong>
                <p>Page: ${location}</p>
                <p id="citation-text">${content}</p>
            </div>
        `;
        if (i == 0) {
            citationDiv.classList.add("active");
        }
        citationItems.push(citationDiv);
        carouselContainer.appendChild(citationDiv);
    }

    // Only add navigation if there are multiple citations
    if (citations.length > 1) {
        // Add Navigation Buttons
        let controls = document.createElement("div");
        controls.className = "carousel-controls";

        let prevButton = document.createElement("button");
        prevButton.className = "carousel-control-prev";
        prevButton.innerHTML = "&#10094;";
        prevButton.onclick = () => showCitationItem(-1);

        let nextButton = document.createElement("button");
        nextButton.className = "carousel-control-next";
        nextButton.innerHTML = "&#10095;";
        nextButton.onclick = () => showCitationItem(1);

        controls.appendChild(prevButton);
        controls.appendChild(nextButton);
        carouselContainer.appendChild(controls);
    }

    document.getElementById("citationModal").style.display = "flex";
}

function closeCitationModal() {
    document.getElementById("citationModal").style.display = "none";
}

function showCitationItem(direction) {
    citationItems[currentCitationIndex].classList.remove("active");
    currentCitationIndex = (currentCitationIndex + direction + citationItems.length) % citationItems.length;
    citationItems[currentCitationIndex].classList.add("active");
}

function addQuickMessages(topic, messages) {
    // Add intro message to the message window
    let introMessage = document.createElement("div");
    introMessage.className = "message bot-message";
    introMessage.innerText = lang_context["quick_messages_intro"];

    document.getElementById("messageWindow").appendChild(introMessage);

    // Function to add quick replies to the message window
    let quickRepliesContainer = document.createElement("div");
    quickRepliesContainer.className = "quick-replies";

    console.log("messages", messages);

    messages.forEach(msg => {
        let button = document.createElement("button");
        button.className = "quick-reply";
        button.innerText = msg;

        // On click, set input field and simulate sending message
        button.onclick = function () {
            document.getElementById("userInput").value = msg;
            sendUserInput();  // Call your function that sends the message
        };

        quickRepliesContainer.appendChild(button);
    });

    // Append the quick replies to the message window
    document.getElementById("messageWindow").appendChild(quickRepliesContainer);

    scrollToBottom();
}
function addQuickTopics(topics) {
    // Add intro message to the message window
    let introMessage = document.createElement("div");
    introMessage.className = "message bot-message";
    introMessage.innerText = lang_context["quick_topics_intro"];

    document.getElementById("messageWindow").appendChild(introMessage);

    // Function to add quick replies to the message window
    let quickRepliesContainer = document.createElement("div");
    quickRepliesContainer.className = "quick-replies";

    console.log("topics", topics);

    // Create a button for each topic
    Object.keys(topics).forEach(topic => {
        console.log("topic", topic);
        let button = document.createElement("button");
        button.className = "quick-reply";
        button.innerText = topic;

        // On click, create a set of message with attached to the topic
        button.onclick = function () {
            addQuickMessages(topic, topics[topic]);
        };
        quickRepliesContainer.appendChild(button);
    });

    // Append the quick replies to the message window
    document.getElementById("messageWindow").appendChild(quickRepliesContainer);
}
async function sendUserInput() {
    // Function that sends user input to the server and awaits a response
    let userInput = document.getElementById('userInput').value.trim(); // Trimmed user input
    let chatBox = document.getElementById('messageWindow'); // Chat box element

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

    scrollToBottom();

    let spinner = document.createElement("div");
    spinner.className = "spinner-border text-dark";
    spinner.role = "status";
    chatBox.appendChild(spinner);

    // Get the current language
    const language = getLanguageFromCookie();

    try {
        const response = await fetch(analyzeUserInputUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({
                message: userInput,
                language: language
            })
        });
        let data = await response.json();
        answer = data.message.answer;
        suggested_prompts = data.suggested_prompts;
        console.log("suggest_prompts", suggested_prompts);

        // Log the full response for debugging
        console.log("Full response:", data);

        // Check if citations exist for each party
        for (let party in answer) {
            console.log(`Party ${party} has ${answer[party].citations_count} citations`);
            console.log(`Citations:`, answer[party].citations);
        }

        // No checks for errors yet
        displayBotMessage(answer);

        // Add prompts
        addQuickTopics(data.suggested_prompts);

    } catch (error) {
        console.error("Error sending message:", error);
    } finally {
        spinner.remove();
        document.getElementById('userInput').value = '';
    }
}

function initPrompts() {
    // Add quick topics to the chat window
    console.log("initPrompts");
    const response = fetch(createInitPrompts, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
    }).then(data => data.json()).then(data => addQuickTopics(data));
}

document.addEventListener("DOMContentLoaded", function () {
    botInitialMessage();
    console.log("initPrompts");
    initPrompts();
});