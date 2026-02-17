
/* Intercept requests to outside URLs and block them until scanned! */
chrome.webNavigation.onBeforeNavigate.addListener((details) => {
    if (details.frameId !== 0) return;
    
    const url = new URL(details.url);

    if (url.protocol === 'chrome-extension:') return;

    // perform check -- we will load our model and use it here!

    chrome.tabs.update(details.tabId, {
        url: chrome.runtime.getURL("html_code/checking.html") + "?target=" + encodeURIComponent(details.url)
    });
});