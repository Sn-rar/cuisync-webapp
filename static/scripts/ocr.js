document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('image-upload-input');
  const imageForm = document.getElementById('image-search-form');
  const originSelect = document.getElementById('img-origin');
  const targetSelect = document.getElementById('img-target');
  const imageSearchButton = imageForm.querySelector('button[type="submit"]');
  const hiddenDishInput = document.getElementById('hidden-dish-name');
  const defaultState = document.getElementById('dropzone-default');
  const previewState = document.getElementById('dropzone-preview');
  const previewImg = document.getElementById('preview-img');
  const fileName = document.getElementById('file-name');
  const removeBtn = document.getElementById('remove-img-btn');
  const selectedDishCard = document.getElementById('selected-dish-card');
  const selectedDishName = document.getElementById('selected-dish-name');
  const selectedDishAltName = document.getElementById('selected-dish-alt-name');
  const selectedDishCountry = document.getElementById('selected-dish-country');
  const selectedDishImage = document.getElementById('selected-dish-image');
  const ocrLoading = document.getElementById('ocr-loading');
  const picker = document.getElementById('dish-picker-overlay');
  const pickerList = document.getElementById('dish-picker-list');
  const pickerClose = document.getElementById('dish-picker-close');
  let activeScanController = null;
  let detectedMatches = [];

  function closeDishPicker() {
    picker.classList.add('hidden');
  }

  function setOcrLoading(isLoading) {
    ocrLoading.style.display = isLoading ? 'block' : 'none';
    originSelect.disabled = isLoading;
    targetSelect.disabled = isLoading;
    imageSearchButton.disabled = isLoading;
    imageForm.classList.toggle('is-ocr-loading', isLoading);
  }

  function selectDish(match, showCard = true, canChange = true) {
    const option = [...originSelect.options].find(
      (item) => item.value.toLowerCase() === match.country.toLowerCase()
    );
    if (option) originSelect.value = option.value;
    hiddenDishInput.value = match.title;
    selectedDishName.textContent = match.title;
    if (match.alternative_title) {
      selectedDishAltName.textContent = match.alternative_title;
      selectedDishAltName.style.display = '';
    } else {
      selectedDishAltName.textContent = '';
      selectedDishAltName.style.display = 'none';
    }
    selectedDishCountry.textContent = match.country;
    if (match.image_link) {
      selectedDishImage.src = match.image_link;
      selectedDishImage.alt = match.title;
      selectedDishImage.style.display = '';
    } else {
      selectedDishImage.removeAttribute('src');
      selectedDishImage.style.display = 'none';
    }
    selectedDishCard.style.display = showCard ? 'flex' : 'none';
    selectedDishCard.querySelector('.selected-dish-change').style.display = canChange ? 'inline' : 'none';
    selectedDishCard.setAttribute(
      'aria-label',
      canChange ? 'Change selected dish' : 'Selected dish'
    );
    closeDishPicker();
  }

  function showDishPicker(matches) {
    detectedMatches = matches;
    if (matches.length === 1) {
      selectDish(matches[0], true, false);
      return;
    }
    pickerList.replaceChildren();

    matches.forEach((match) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'dish-picker-option';

      const infoDiv = document.createElement('div');
      infoDiv.className = 'dish-picker-info';

      const titleEl = document.createElement('strong');
      titleEl.textContent = match.title;
      infoDiv.appendChild(titleEl);

      if (match.alternative_title) {
        const altEl = document.createElement('em');
        altEl.className = 'dish-picker-alt-name';
        altEl.textContent = match.alternative_title;
        infoDiv.appendChild(altEl);
      }

      const countryEl = document.createElement('span');
      countryEl.className = 'dish-picker-country';
      countryEl.textContent = match.country;
      infoDiv.appendChild(countryEl);

      button.appendChild(infoDiv);

      const imgEl = document.createElement('img');
      imgEl.className = 'dish-picker-image';
      imgEl.alt = match.title;
      imgEl.loading = 'lazy';

      if (match.image_link) {
        imgEl.src = match.image_link;
      } else {
        imgEl.style.display = 'none';
      }

      imgEl.addEventListener('error', () => {
        imgEl.style.display = 'none';
      });

      button.appendChild(imgEl);

      button.addEventListener('click', (e) => {
        e.stopPropagation();
        selectDish(match);
      });
      pickerList.appendChild(button);
    });

    picker.classList.remove('hidden');
  }

  async function extractImageInfo(file) {
    hiddenDishInput.value = '';
    closeDishPicker();
    selectedDishCard.style.display = 'none';
    detectedMatches = [];
    if (activeScanController) activeScanController.abort();
    const scanController = new AbortController();
    activeScanController = scanController;
    const formData = new FormData();
    formData.append('dish_image', file);
    formData.append('origin_country', originSelect.value);
    setOcrLoading(true);

    try {
      const response = await fetch('/extract-dish-info', {
        method: 'POST',
        body: formData,
        signal: scanController.signal,
      });
      const data = await response.json();
      if (activeScanController !== scanController) return;

      if (data.matches && data.matches.length) {
        const detectedCountries = [...new Set(
          data.matches.map((match) => match.country.toLowerCase())
        )];
        if (detectedCountries.length === 1) {
          const option = [...originSelect.options].find(
            (item) => item.value.toLowerCase() === detectedCountries[0]
          );
          if (option) originSelect.value = option.value;
        }
        showDishPicker(data.matches);
      } else {
        const dishName = data.raw_text
          ? data.raw_text
              .split(/\r?\n/)
              .map((line) => line.trim())
              .filter(Boolean)
              .join(' ')
          : 'this dish';

        showErrorModal(
          "We Can't Find Your Dish",
          `We couldn't find "${dishName}" in the recipe database. Try another dish or use a clearer photo.`
        );
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Error connecting to extraction service:', err);
      showErrorModal('Scanning Error', 'The menu could not be scanned. Please try the photo again.');
    } finally {
      if (activeScanController === scanController) {
        activeScanController = null;
        setOcrLoading(false);
      }
    }
  }

  function displayPreview(file) {
    if (!file.type.startsWith('image/')) {
      showErrorModal('Unsupported File', 'Please upload an image file.');
      return;
    }
    originSelect.value = '';
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImg.src = event.target.result;
      fileName.textContent = file.name;
      defaultState.style.display = 'none';
      previewState.style.display = 'flex';
    };
    reader.readAsDataURL(file);
    extractImageInfo(file);
  }

  // Prevent fileInput from opening when clicking removeBtn or selectedDishCard
  dropzone.addEventListener('click', (event) => {
    if (event.target.closest('#remove-img-btn') || event.target.closest('#selected-dish-card')) {
      return;
    }
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) displayPreview(fileInput.files[0]);
  });

  ['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  }));

  ['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  }));

  dropzone.addEventListener('drop', (event) => {
    const [file] = event.dataTransfer.files;
    if (file) {
      fileInput.files = event.dataTransfer.files;
      displayPreview(file);
    }
  });

  originSelect.addEventListener('change', () => {
    if (fileInput.files[0]) extractImageInfo(fileInput.files[0]);
  });

  pickerClose.addEventListener('click', closeDishPicker);

  picker.addEventListener('click', (event) => {
    if (event.target === picker) closeDishPicker();
  });

  // Stop bubbling to dropzone so file upload does not open
  selectedDishCard.addEventListener('click', (event) => {
    event.stopPropagation();
    if (detectedMatches.length >= 2) {
      showDishPicker(detectedMatches);
    }
  });

  removeBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    if (activeScanController) activeScanController.abort();
    setOcrLoading(false);
    fileInput.value = '';
    previewImg.src = '';
    fileName.textContent = '';
    hiddenDishInput.value = '';
    selectedDishCard.style.display = 'none';
    detectedMatches = [];
    closeDishPicker();
    previewState.style.display = 'none';
    defaultState.style.display = 'block';
  });

  imageForm.addEventListener('submit', (event) => {
    if (fileInput.files[0] && !hiddenDishInput.value) {
      event.preventDefault();
      showErrorModal('Choose a Dish', 'Select one of the detected menu dishes before starting the comparison.');
    }
  });
});