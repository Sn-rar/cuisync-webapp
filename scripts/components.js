class AppNav extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <header class="navbar" id="navbar">
        <a href="index.html"><div class="logo">CUISYNC</div></a>
        
        <button class="hamburger-toggle" id="hamburger-btn" aria-label="Toggle Navigation Menu">
          <span class="bar"></span>
          <span class="bar"></span>
          <span class="bar"></span>
        </button>

        <nav class="nav-links" id="nav-menu">
          <a href="index.html">HOME</a>
          <a href="analysis.html">ANALYSIS</a>
          <a href="faqs.html">FAQS</a>
        </nav>
      </header>
      <div class="nav-overlay" id="nav-overlay"></div>
    `;

    // Initialize navbar scripts immediately after rendering
    this.initNavigation();
  }

  initNavigation() {
    const navbar = this.querySelector('#navbar');
    const hamburgerBtn = this.querySelector('#hamburger-btn');
    const navMenu = this.querySelector('#nav-menu');
    const navOverlay = this.querySelector('#nav-overlay');

    // Sticky Scroll Effect
    window.addEventListener('scroll', () => {
      if (navbar) {
        if (window.scrollY > 40) {
          navbar.classList.add('scrolled');
        } else {
          navbar.classList.remove('scrolled');
        }
      }
    });

    // Mobile Menu Drawer Toggle
    if (hamburgerBtn && navMenu && navOverlay) {
      const toggleMenu = () => {
        hamburgerBtn.classList.toggle('open');
        navMenu.classList.toggle('active');
        navOverlay.classList.toggle('active');
        document.body.style.overflow = navMenu.classList.contains('active') ? 'hidden' : 'auto';
      };

      hamburgerBtn.addEventListener('click', toggleMenu);
      navOverlay.addEventListener('click', toggleMenu);

      this.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
          if (navMenu.classList.contains('active')) {
            toggleMenu();
          }
        });
      });
    }
  }
}

class AppFooter extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <footer class="footer">
        <div class="footer-container">
          
          <div class="footer-col footer-brand">
            <h3 class="footer-logo">CUISYNC</h3>
            <p>
              Developed for academic research on Asian cuisine similarities. 
              This evidence-based platform is AI-powered; content is for 
              informational and research purposes only.
            </p>
            <div class="brand-underline"></div>
          </div>

          <div class="footer-col">
            <h4 class="footer-heading">EXPLORE</h4>
            <ul class="footer-links">
              <li><a href="index.html">Home</a></li>
              <li><a href="analysis.html">Analysis</a></li>
              <li><a href="#">FAQS</a></li>
            </ul>
          </div>

          <div class="footer-col">
            <h4 class="footer-heading">ASIAN CUISINES</h4>
            <div class="cuisine-grid">
              <ul>
                <li><a href="analysis.html?country=cambodia">Cambodia</a></li>
                <li><a href="analysis.html?country=china">China</a></li>
                <li><a href="analysis.html?country=india">India</a></li>
                <li><a href="analysis.html?country=indonesia">Indonesia</a></li>
                <li><a href="analysis.html?country=japan">Japan</a></li>
              </ul>
              <ul>
                <li><a href="analysis.html?country=malaysia">Malaysia</a></li>
                <li><a href="analysis.html?country=myanmar">Myanmar</a></li>
                <li><a href="analysis.html?country=philippines">Philippines</a></li>
                <li><a href="analysis.html?country=saudi-arabia">Saudi Arabia</a></li>
                <li><a href="analysis.html?country=south-korea">South Korea</a></li>
              </ul>
              <ul>
                <li><a href="analysis.html?country=turkey">Turkey</a></li>
                <li><a href="analysis.html?country=thailand">Thailand</a></li>
                <li><a href="analysis.html?country=vietnam">Vietnam</a></li>
              </ul>
            </div>
          </div>
        </div>

        <div class="footer-bottom">
          <p>© 2025 <span>Cuisync</span>. All rights reserved.</p>
        </div>
      </footer>
    `;
  }
}

customElements.define('app-nav', AppNav);
customElements.define('app-footer', AppFooter);

  //===========================================================================
  // STICKY NAVBAR
  //===========================================================================
  window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (navbar) {
      if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    }
  });