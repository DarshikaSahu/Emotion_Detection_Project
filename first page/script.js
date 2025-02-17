function toggleForm() {
    var loginForm = document.getElementById('login-form');
    var signupForm = document.getElementById('signup-form');
    if (loginForm.style.display === 'none') {
        loginForm.style.display = 'block';
        signupForm.style.display = 'none';
    } else {
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
    }
}

// Toggle password visibility
function togglePassword(formType) {
    let passwordField;
    if (formType === 'login') {
        passwordField = document.getElementById('login-password');
    } else if (formType === 'signup') {
        passwordField = document.getElementById('signup-password');
    } else if (formType === 'signup-confirm') {
        passwordField = document.getElementById('signup-confirm-password');
    }

    if (passwordField.type === "password") {
        passwordField.type = "text";
    } else {
        passwordField.type = "password";
    }
}

// Login form validation
document.getElementById('login-form').addEventListener('submit', function (event) {
    event.preventDefault();
    var username = document.getElementById('login-username').value;
    var password = document.getElementById('login-password').value;
    if (username && password) {
        alert('Login Successful');
    } else {
        alert('Please fill in all fields');
    }
});

// Sign-up form validation
document.getElementById('signup-form').addEventListener('submit', function (event) {
    event.preventDefault();
    var username = document.getElementById('signup-username').value;
    var email = document.getElementById('signup-email').value;
    var password = document.getElementById('signup-password').value;
    var confirmPassword = document.getElementById('signup-confirm-password').value;

    if (username && email && password && confirmPassword) {
        if (password === confirmPassword) {
            alert('Sign Up Successful');
        } else {
            alert('Passwords do not match');
        }
    } else {
        alert('Please fill in all fields');
    }
});
