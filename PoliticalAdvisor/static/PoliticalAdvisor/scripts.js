function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

document.getElementById('userInput').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default form submission
        document.querySelector('button[onclick="sendUserInput()"]').click(); // Trigger the button click
    }
});

function scrollToBottom() {
    const messageWindow = document.getElementById('messageWindow');
    messageWindow.scrollTop = messageWindow.scrollHeight;
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
        botMessage.appendChild(createTable(data));
    }

    // show bot message
    document.getElementById('messageWindow').appendChild(botMessage);
    scrollToBottom();

}

function displayProgCitations(data) {
    let botMessage = document.createElement("div");
    botMessage.className = "message bot-message";

    // Set the inner text depending on format of the data
    if (typeof data === "string") {
        botMessage.innerText = data;
    } else if (typeof data === "object") {
        botMessage.appendChild(createProgCitations(data));
    }

    // show bot message
    document.getElementById('messageWindow').appendChild(botMessage);
    scrollToBottom();
}

function createProgCitations(data) {
    let table = document.createElement("table");
    table.style.width = "100%";
    table.style.borderCollapse = "collapse";
    table.style.marginTop = "10px";
    table.style.verticalAlign = "top";

    // Create table header
    let thead = table.createTHead();
    let headerRow = thead.insertRow();
    let headers = ["Program", "Seite", "Zitat"];
    headers.forEach(headerText => {
        let th = document.createElement("th");
        th.innerText = headerText;
        th.style.border = "1px solid #ddd";
        th.style.padding = "8px";
        headerRow.appendChild(th);
    });

    let tbody = table.createTBody();
    let rows =[];

    for (let citat in data) {
        let row = tbody.insertRow();

        // Partei Name
        let cell1 = row.insertCell(0);
        cell1.innerHTML = `<strong>${data[citat].source.toUpperCase()}</strong>`;
        cell1.style.border = "1px solid #ddd";
        cell1.style.padding = "8px";
        cell1.style.verticalAlign = "top";

        // Seite
        let cell2 = row.insertCell(1);
        cell2.innerText = `${data[citat].location}`;
        cell2.style.textAlign = "center";
        cell2.style.border = "1px solid #ddd";
        cell2.style.padding = "8px";
        cell2.style.verticalAlign = "top";

        // Zitat
        let cell3 = row.insertCell(2);
        cell3.innerText = data[citat].content;
        cell3.style.border = "1px solid #ddd";
        cell3.style.padding = "8px";
        cell2.style.verticalAlign = "top";

        rows.push(row);
    }

    // Append sorted rows to tbody
    rows.forEach(row => tbody.appendChild(row));

    return table;



}

function createTable(data) {
    let table = document.createElement("table");
    table.style.width = "100%";
    table.style.borderCollapse = "collapse";
    table.style.marginTop = "10px";

    // Create table header
    let thead = table.createTHead();
    let headerRow = thead.insertRow();
    let headers = ["Partei", "Relevanz", "Position"];
    headers.forEach(headerText => {
        let th = document.createElement("th");
        th.innerText = headerText;
        th.style.border = "1px solid #ddd";
        th.style.padding = "8px";
        headerRow.appendChild(th);
    });

    // Create table body
    let tbody = table.createTBody();
    let rows =[];

    for (let party in data) {
        console.log("party", party);
        let row = tbody.insertRow();

        // Partei Name
        let cell1 = row.insertCell(0);
        cell1.innerHTML = `<strong>${party.toUpperCase()}</strong>`;
        cell1.style.border = "1px solid #ddd";
        cell1.style.padding = "8px";

        // Relevanz
        let cell2 = row.insertCell(1);
        cell2.innerText = `${data[party].agreement}%`;
        cell2.style.textAlign = "center";
        cell2.style.border = "1px solid #ddd";
        cell2.style.padding = "8px";

        // Position
        let cell3 = row.insertCell(2);
        cell3.innerText = data[party].explanation;
        cell3.style.border = "1px solid #ddd";
        cell3.style.padding = "8px";

        rows.push(row);
    }

    // Sort rows by Zustimmung column
    rows.sort((a, b) => {
        let aValue = parseFloat(a.cells[1].innerText);
        let bValue = parseFloat(b.cells[1].innerText);
        return bValue - aValue;
    });

    // Append sorted rows to tbody
    rows.forEach(row => tbody.appendChild(row));

    return table;
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
        displayProgCitations(prog_citations);


    } catch (error) {
        console.error("Error sending message:", error);
    }
}