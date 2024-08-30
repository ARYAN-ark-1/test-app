document.addEventListener("DOMContentLoaded", function() {
    const moreBtn = document.querySelector('.more-btn');
    const hiddenServices = document.querySelectorAll('.service-item.hidden');

    moreBtn.addEventListener('click', function() {
        hiddenServices.forEach(service => {
            service.classList.toggle('hidden'); // Toggle hidden class
        });
        
        // Change button text based on visibility
        if (moreBtn.textContent === 'Show More Companies') {
            moreBtn.textContent = 'Show Less Companies';
        } else {
            moreBtn.textContent = 'Show More Companies';
        }
    });
});
