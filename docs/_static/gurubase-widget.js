document.addEventListener('DOMContentLoaded', function() {
    // Customize widget settings
    const widgetSettings = {
        widgetId: "c4FTi9Br1lRBsvAlA63F-L7W_gfkMOlxnRN9BahgfZY", // local widget ID
        // widgetId: "4wUO7oVdkTYSX2X8Nx3s6YUh6K3ISE1Iu4-zm4DWoHY",
        text: "Ask AI", // Optional - Button text
        margins: { bottom: "20px", right: "20px" }, // Optional
        lightMode: "dark", // Optional - Force light mode
        // bgColor: "YOUR_BG_COLOR", // Optional - Widget background color
        // iconUrl: "YOUR_ICON_URL", // Optional - Widget icon URL
        name: "CONAN", // Optional - Widget name
        overlapContent: "false" // Optional - Whether to overlap the main content or shrink its width with the sidebar
    };

    // Load the GuruBase widget
    const guruScript = document.createElement("script");
    guruScript.src = "https://widget.gurubase.io/widget.latest.min.js";
    guruScript.defer = true;
    guruScript.id = "guru-widget-id";

    // Add widget settings as data attributes
    Object.entries({
        "data-widget-id": widgetSettings.widgetId,
        "data-text": widgetSettings.text,
        "data-margins": JSON.stringify(widgetSettings.margins),
        "data-light-mode": widgetSettings.lightMode,
        "data-bg-color": widgetSettings.bgColor,
        "data-icon-url": widgetSettings.iconUrl,
        "data-name": widgetSettings.name,
        "data-overlap-content": widgetSettings.overlapContent
    }).forEach(([key, value]) => {
        guruScript.setAttribute(key, value);
    });

    // Append the script to the document
    document.body.appendChild(guruScript);
});