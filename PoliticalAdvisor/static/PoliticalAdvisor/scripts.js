function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

document.getElementById('userInput').addEventListener('keypress', function(event) {
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
        lastChild.scrollIntoView({behavior: "smooth", block: "start"});
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
        let button = document.createElement("button");
        button.innerText = "Citations";
        button.className = "btn btn-dark btn-sm";
        button.onclick = function() {
            openCitationModal(party, data[party].citations);
        };

        header.innerHTML = `
            <strong>${party.toUpperCase()}</strong> 
            <span class="ml-2">Relev.: ${data[party].agreement}%</span>
        `;
        header.appendChild(button);

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



function showCarouselItem(direction) {
    // Get all carousel items
    let items = document.querySelectorAll(".carousel-item");
    // Remove active class from current item
    items[currentCarouselIndex].classList.remove("active");
    // Calculate new index
    currentCarouselIndex = (currentCarouselIndex + direction + items.length) % items.length;
    // Add active class to new item
    items[currentCarouselIndex].classList.add("active");

    scrollToBottom();
}





function openCitationModal(party, citations) {
    document.getElementById("modal-title").innerText = `Citations for ${party.toUpperCase()}`;
    let carouselContainer = document.getElementById("citationCarousel");
    carouselContainer.innerHTML = "";  // Clear existing citations

    citationItems = [];  // Reset citations
    currentCitationIndex = 0;

    // Create citation elements
    for (let i in citations) {
        let citationDiv = document.createElement("div");
        citationDiv.className = "carousel-item";
        citationDiv.innerHTML = `
            <div class="citation">
                <strong>${citations[i].source.toUpperCase()}</strong>
                <p>Page: ${citations[i].location}</p>
                <p id="citation-text">${citations[i].content}</p>
            </div>
        `;
        if (i == 0) {
            citationDiv.classList.add("active");
        }
        citationItems.push(citationDiv);
        carouselContainer.appendChild(citationDiv);
    }

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
    spinner.role="status";
    chatBox.appendChild(spinner);


    try {
        const response = await fetch(analyzeUserInputUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({ message: userInput })
        });
        let data = await response.json();
        answer = data.message.answer;
        prog_citations = data.message.citations;


        console.log("prog_citations", prog_citations);

        // No checks for errors yet
        displayBotMessage(answer);

    } catch (error) {
        console.error("Error sending message:", error);
    } finally {
        spinner.remove();
    }

}