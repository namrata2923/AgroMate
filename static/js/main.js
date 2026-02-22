/* ================================================================
   AgroMate – main.js
================================================================ */

document.addEventListener('DOMContentLoaded', function () {

  // ─── AOS Initialise ────────────────────────────────────────────
  AOS.init({
    duration: 700,
    once: true,
    offset: 80,
    easing: 'ease-out-cubic'
  });

  // ─── Navbar scroll effect ──────────────────────────────────────
  const navbar = document.getElementById('mainNav');
  if (navbar) {
    window.addEventListener('scroll', () => {
      navbar.classList.toggle('scrolled', window.scrollY > 50);
    });
  }

  // ─── Back to top ───────────────────────────────────────────────
  const btt = document.getElementById('backToTop');
  if (btt) {
    window.addEventListener('scroll', () => {
      btt.classList.toggle('visible', window.scrollY > 400);
    });
    btt.addEventListener('click', e => {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // ─── Counter animation ─────────────────────────────────────────
  const counters = document.querySelectorAll('.counter-number');
  if (counters.length) {
    const animateCounter = (el) => {
      const target   = parseInt(el.dataset.target, 10);
      const duration = 2000;
      const step     = target / (duration / 16);
      let current    = 0;
      const timer    = setInterval(() => {
        current += step;
        if (current >= target) {
          el.textContent = target.toLocaleString();
          clearInterval(timer);
        } else {
          el.textContent = Math.floor(current).toLocaleString();
        }
      }, 16);
    };

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });

    counters.forEach(c => observer.observe(c));
  }

  // ─── Form loading state ────────────────────────────────────────
  document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function () {
      const btn = this.querySelector('[type="submit"]');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analysing…`;
      }
    });
  });

  // ─── Smooth scroll for anchor links ────────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ─── Tooltip init (Bootstrap) ──────────────────────────────────
  const tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltipEls.forEach(el => new bootstrap.Tooltip(el));

  // ─── Active nav link highlight ─────────────────────────────────
  const currentPath = window.location.pathname;
  document.querySelectorAll('.site-navbar .nav-link').forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });

  console.log('🌱 AgroMate JS loaded successfully');
});
