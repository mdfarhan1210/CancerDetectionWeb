document.addEventListener("DOMContentLoaded", () => {
    function chatOpen() {
        document.getElementById("chat-open").style.display = "none";
        document.getElementById("chat-close").style.display = "block";
        document.getElementById("chat-window1").style.display = "block";
    }

    function chatClose() {
        document.getElementById("chat-open").style.display = "block";
        document.getElementById("chat-close").style.display = "none";
        document.getElementById("chat-window1").style.display = "none";
        document.getElementById("chat-window2").style.display = "none";
    }

    function openConversation() {
        document.getElementById("chat-window2").style.display = "block";
        document.getElementById("chat-window1").style.display = "none";
    }

    function userResponse() {
        let userText = document.getElementById("textInput").value;
        console.log("User Input:", userText);
        console.log("Trimmed Input:", userText.trim());
        if (!userText.trim()) {
            alert("Please type something!");
            return;
        }

        document.getElementById("messageBox").innerHTML += `
            <div class="first-chat">
                <p>${userText}</p>
                <div class="arrow"></div>
            </div>`;

        document.getElementById("textInput").value = "";
        document.getElementById("messageBox").scrollTop = document.getElementById("messageBox").scrollHeight;

        setTimeout(() => {
            adminResponse(userText);
        }, 1000);
    }

    async function adminResponse(userText) {
        fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userText })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("messageBox").innerHTML += `
                <div class="second-chat">
                    <div class="circle"></div>
                    <p>${data.response}</p>
                    <div class="arrow"></div>
                </div>`;

            document.getElementById("messageBox").scrollTop = document.getElementById("messageBox").scrollHeight;
        })
        .catch(error => console.log("Error:", error));
    }

    document.getElementById("textInput").addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            e.preventDefault(); 
            userResponse();
        }
    });

    // Expose functions to global scope
    window.openConversation = openConversation;
    window.chatOpen = chatOpen;
    window.chatClose = chatClose;
    window.userResponse = userResponse;
});
