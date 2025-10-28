// Add copy button to code blocks
document.addEventListener('DOMContentLoaded', function() {
  // Find all code blocks
  const codeBlocks = document.querySelectorAll('pre');

  codeBlocks.forEach(function(codeBlock) {
    // Skip if already has a copy button
    if (codeBlock.parentElement.querySelector('.copy-button')) {
      return;
    }

    // Create copy button
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.innerHTML = `
      <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">
        <path fill="currentColor" d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z" />
      </svg>
      <svg class="check-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" style="display: none;">
        <path fill="currentColor" d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z" />
      </svg>
    `;
    button.setAttribute('title', 'Copy code');

    // Add click event
    button.addEventListener('click', function() {
      const code = codeBlock.querySelector('code');
      const text = code ? code.innerText : codeBlock.innerText;

      // Copy to clipboard
      navigator.clipboard.writeText(text).then(function() {
        // Show success feedback
        const copyIcon = button.querySelector('.copy-icon');
        const checkIcon = button.querySelector('.check-icon');

        copyIcon.style.display = 'none';
        checkIcon.style.display = 'inline';
        button.classList.add('copied');

        // Reset after 2 seconds
        setTimeout(function() {
          copyIcon.style.display = 'inline';
          checkIcon.style.display = 'none';
          button.classList.remove('copied');
        }, 2000);
      }).catch(function(err) {
        console.error('Failed to copy code: ', err);
      });
    });

    // Create wrapper if needed
    let wrapper = codeBlock.parentElement;
    if (!wrapper.classList.contains('highlight')) {
      wrapper = document.createElement('div');
      wrapper.className = 'highlight';
      codeBlock.parentNode.insertBefore(wrapper, codeBlock);
      wrapper.appendChild(codeBlock);
    }

    // Add button to wrapper
    wrapper.style.position = 'relative';
    wrapper.appendChild(button);
  });
});

