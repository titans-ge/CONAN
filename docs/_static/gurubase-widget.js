document.addEventListener('DOMContentLoaded', function() {
    // Customize widget settings
    const widgetSettings = {
        widgetId: "4wUO7oVdkTYSX2X8Nx3s6YUh6K3ISE1Iu4-zm4DWoHY", // Replace with your widget ID
        text: "Ask AI", // Optional - Button text
        margins: { bottom: "20px", right: "20px" }, // Optional
        lightMode: "dark", // Optional - Force light mode
        // bgColor: "YOUR_BG_COLOR", // Optional - Widget background color
        // iconUrl: "YOUR_ICON_URL", // Optional - Widget icon URL
        name: "CONAN", // Optional - Widget name
        overlapContent: "false" // Optional - Whether to overlap the main content or shrink its width with the sidebar
    };

    // Load marked.js
    const markedScript = document.createElement('script');
    markedScript.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    markedScript.async = true;
    document.body.appendChild(markedScript);

    // Load the GuruBase widget
    markedScript.onload = () => {
        const guruScript = document.createElement("script");
        guruScript.src = "https://widget.gurubase.io/widget.latest.min.js";
        // guruScript.defer = true;
        guruScript.async = true;
        guruScript.id = "guru-widget-id";
        guruScript.setAttribute("data-widget-id", widgetSettings.widgetId);
        guruScript.setAttribute("data-text", widgetSettings.text);
        guruScript.setAttribute("data-margins", JSON.stringify(widgetSettings.margins));
        guruScript.setAttribute("data-light-mode", widgetSettings.lightMode);
        // guruScript.setAttribute("data-bg-color", widgetSettings.bgColor);
        // guruScript.setAttribute("data-icon-url", widgetSettings.iconUrl);
        guruScript.setAttribute("data-name", widgetSettings.name);
        guruScript.setAttribute("data-overlap-content", widgetSettings.overlapContent); 
        guruScript.setAttribute("data-marked", "true"); // Enable marked.js support
        guruScript.setAttribute("data-marked-options", JSON.stringify({
            gfm: true,
            breaks: true,
            smartLists: true,
            smartypants: true
        }));
        document.body.appendChild(guruScript);
    };

    // // Add widget settings as data attributes
    // Object.entries({
    //     "data-widget-id": widgetSettings.widgetId,
    //     "data-text": widgetSettings.text,
    //     "data-margins": JSON.stringify(widgetSettings.margins),
    //     "data-light-mode": widgetSettings.lightMode,
    //     "data-bg-color": widgetSettings.bgColor,
    //     "data-icon-url": widgetSettings.iconUrl,
    //     "data-name": widgetSettings.name,
    //     "data-overlap-content": widgetSettings.overlapContent
    // }).forEach(([key, value]) => {
    //     guruScript.setAttribute(key, value);
    // });

    // // Append the script to the document
    // document.body.appendChild(guruScript);
});