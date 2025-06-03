function togglePasswordVisibility() {
    var passwordInput = document.getElementById('password');
    var toggleIcon = document.getElementById('password-icon');
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        toggleIcon.name = 'eye-off-outline';  // Change to eye-off icon
    } else {
        passwordInput.type = 'password';
        toggleIcon.name = 'eye-outline';  // Change back to eye icon
    }
}
