document.addEventListener('DOMContentLoaded', function () {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    const submitBtn = document.getElementById('submitBtn');

    // Function to update the submit button state
    function updateSubmitButtonState() {
        const isAnyCheckboxChecked = Array.from(checkboxes).some(checkbox => checkbox.checked);
        submitBtn.disabled = !isAnyCheckboxChecked;
    }

    // Function to set the initial state and attach event listeners
    checkboxes.forEach(checkbox => {
        const input = checkbox.closest('.checkbox-field').querySelector('input[type="text"], input[type="email"], input[type="password"], input[type="file"]');
        // Set initial state based on checkbox
        input.disabled = !checkbox.checked;
        if (!checkbox.checked) {
            if (input.type === 'file') {
                // Resetting file input more reliably
                input.value = ''; // This is the only reliable way to clear file inputs in modern browsers
            } else {
                input.value = '';
            }
        }

        // Add event listener for changes
        checkbox.addEventListener('change', () => {
            input.disabled = !checkbox.checked;
            if (!checkbox.checked) {
                if (input.type === 'file') {
                    input.value = ''; // Clear file input
                } else {
                    input.value = ''; // Clear other types of inputs
                }
            }
            // Update the state of the submit button
            updateSubmitButtonState();
        });
    });

    // Initial check to set the correct state of the submit button
    updateSubmitButtonState();
});
