/**
 * Fix for MkDocs search: Don't show "No results found" when search is cleared
 * This script overrides the default search behavior to improve UX
 */

document.addEventListener('DOMContentLoaded', function() {
  // Wait for search to be initialized
  setTimeout(function() {
    var searchInput = document.getElementById('mkdocs-search-query');
    if (!searchInput) return;

    // Store the original doSearch function if it exists
    var originalDoSearch = window.doSearch;

    // Override the doSearch function
    window.doSearch = function() {
      var query = searchInput.value.trim();
      var searchResults = document.getElementById('mkdocs-search-results');

      if (!searchResults) {
        // If search results element doesn't exist yet, call original function
        if (originalDoSearch) originalDoSearch();
        return;
      }

      // If query is empty or only whitespace, clear results without showing error
      if (query.length === 0) {
        while (searchResults.firstChild) {
          searchResults.removeChild(searchResults.firstChild);
        }
        return;
      }

      // Otherwise, call the original search function
      if (originalDoSearch) {
        originalDoSearch();
      }
    };

    // Also override the displayResults function to handle empty queries
    var originalDisplayResults = window.displayResults;

    window.displayResults = function(results) {
      var searchResults = document.getElementById('mkdocs-search-results');
      var query = searchInput ? searchInput.value.trim() : '';

      if (!searchResults) {
        if (originalDisplayResults) originalDisplayResults(results);
        return;
      }

      // Clear existing results
      while (searchResults.firstChild) {
        searchResults.removeChild(searchResults.firstChild);
      }

      // If query is empty, don't show anything (not even "no results")
      if (query.length === 0) {
        return;
      }

      // If there are results, display them
      if (results && results.length > 0) {
        for (var i = 0; i < results.length; i++) {
          var result = results[i];
          if (window.formatResult) {
            var html = window.formatResult(result.location, result.title, result.summary);
            searchResults.insertAdjacentHTML('beforeend', html);
          }
        }
      } else {
        // Only show "no results" if user actually typed something
        var noResultsText = searchResults.getAttribute('data-no-results-text') || "No results found";
        searchResults.insertAdjacentHTML('beforeend', '<p>' + noResultsText + '</p>');
      }
    };

    // Re-attach the event listener with our new function
    searchInput.removeEventListener("keyup", originalDoSearch);
    searchInput.addEventListener("keyup", window.doSearch);

  }, 500); // Wait 500ms for MkDocs search to initialize
});

