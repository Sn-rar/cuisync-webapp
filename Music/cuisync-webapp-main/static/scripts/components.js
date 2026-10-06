class AppNav extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <header class="navbar" id="navbar">
        <a href="/home"><div class="logo">CUISYNC</div></a>
        
        <button class="hamburger-toggle" id="hamburger-btn" aria-label="Toggle Navigation Menu" aria-controls="nav-menu" aria-expanded="false">
          <span class="hamburger-line"></span>
          <span class="hamburger-line"></span>
          <span class="hamburger-line"></span>
        </button>

        <nav class="nav-links" id="nav-menu">
          <a href="/home">HOME</a>
          <a href="/analysis">ANALYSIS</a>
          <a href="/faqs">FAQS</a>
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

    // Sticky Scroll Effect (also runs once on load in case the page starts scrolled)
    const onScroll = () => {
      navbar && navbar.classList.toggle('scrolled', window.scrollY > 40);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    // Mobile Menu Drawer Toggle
    if (hamburgerBtn && navMenu && navOverlay) {
      const setMenu = (open) => {
        hamburgerBtn.classList.toggle('open', open);
        navMenu.classList.toggle('active', open);
        navOverlay.classList.toggle('active', open);
        hamburgerBtn.setAttribute('aria-expanded', String(open));
        // Class on <html> instead of inline body styles: works on iOS Safari too
        document.documentElement.classList.toggle('menu-open', open);
      };
      const toggleMenu = () => setMenu(!navMenu.classList.contains('active'));

      hamburgerBtn.addEventListener('click', toggleMenu);
      navOverlay.addEventListener('click', () => setMenu(false));

      this.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => setMenu(false));
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') setMenu(false);
      });

      // Close the drawer if the viewport grows past the mobile breakpoint
      window.matchMedia('(min-width: 821px)').addEventListener('change', (e) => {
        if (e.matches) setMenu(false);
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
              <li><a href="/index">Home</a></li>
              <li><a href="/analysis">Analysis</a></li>
              <li><a href="/faqs">FAQS</a></li>
            </ul>
          </div>

          <div class="footer-col">
            <h4 class="footer-heading">DATASET COVERAGE</h4>
            <p class="footer-dataset-desc">
              Our research dataset analyzes recipes across Asian cuisines, incorporating culinary data from 
              Cambodia, China, India, Indonesia, Japan, Malaysia, Myanmar, Philippines, Saudi Arabia, 
              South Korea, Thailand, Turkey, and Vietnam.
            </p>
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