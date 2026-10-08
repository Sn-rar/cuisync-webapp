document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('image-upload-input');
  const imageForm = document.getElementById('image-search-form');
  const originSelect = document.getElementById('img-origin');
  const originDisplay = document.getElementById('img-origin-display');
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
  const selectDishPrompt = document.getElementById('select-dish-prompt');
  const selectDishPromptCount = document.getElementById('select-dish-prompt-count');
  let activeScanController = null;
  let detectedMatches = [];

  // Card under the photo that reopens the picker when several dishes
  // were found but the user closed the picker without choosing one.
  function updateSelectDishPrompt() {
    const pickerOpen = !picker.classList.contains('hidden');
    const needsChoice = detectedMatches.length >= 2 && !hiddenDishInput.value;

    if (needsChoice && !pickerOpen) {
      selectDishPromptCount.textContent =
        `${detectedMatches.length} dishes found · Click to choose`;
      selectDishPrompt.style.display = 'flex';
    } else {
      selectDishPrompt.style.display = 'none';
    }
  }

  function closeDishPicker() {
    picker.classList.add('hidden');
    // Closed (X, Escape, outside click) without choosing a dish:
    // keep the origin empty until a dish is selected.
    if (!hiddenDishInput.value) setOrigin('');
    updateSelectDishPrompt();
  }

  // The origin box isn't a dropdown anymore. It just shows the country of
  // the dish found in the photo, or a placeholder while there isn't one.
  let isScanning = false;

  function updateOriginDisplay() {
    const option = originSelect.options[originSelect.selectedIndex];
    const hasCountry = Boolean(originSelect.value && option);

    if (isScanning) {
      originDisplay.textContent = 'Finding Origin Country';
    } else if (hasCountry) {
      originDisplay.textContent = option.textContent;
    } else {
      originDisplay.textContent = 'Country of Origin';
    }

    originDisplay.classList.toggle('is-placeholder', isScanning || !hasCountry);
    originDisplay.classList.toggle('is-scanning', isScanning);
  }

  function setOrigin(country) {
    const option = [...originSelect.options].find(
      (item) => item.value && item.value.toLowerCase() === (country || '').toLowerCase()
    );
    originSelect.value = option ? option.value : '';
    updateOriginDisplay();
  }

  function setOcrLoading(isLoading) {
    isScanning = isLoading;
    updateOriginDisplay();
    ocrLoading.style.display = isLoading ? 'block' : 'none';
    targetSelect.disabled = isLoading;
    imageSearchButton.disabled = isLoading;
    imageForm.classList.toggle('is-ocr-loading', isLoading);
  }

  function selectDish(match, showCard = true, canChange = true) {
    setOrigin(match.country);
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
    updateSelectDishPrompt();
  }

  async function extractImageInfo(file) {
    hiddenDishInput.value = '';
    detectedMatches = [];
    closeDishPicker();
    selectedDishCard.style.display = 'none';
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
        // The origin stays empty until the user picks a dish
        // (a single match is picked automatically in showDishPicker).
        showDishPicker(data.matches);
      } else if (data.error === 'unsupported_script') {
        // Khmer, Burmese or unreadable text (see ocr.py)
        showErrorModal('Language Not Supported or Blurry Image', data.message);
      } else if (data.error === 'too_large') {
        // Bigger than the upload limit in app.py (MAX_CONTENT_LENGTH)
        showErrorModal('Image Too Large', data.message);
      } else if (data.error === 'no_text') {
        // No text found at all in the photo (see app.py)
        showErrorModal('No Text Detected', data.message);
      } else if (!data.raw_text && data.message) {
        // Something went wrong before any text was read (for example the
        // OCR service on Hugging Face didn't answer). Show the real reason.
        showErrorModal('Scanning Error', data.message);
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
    setOrigin('');
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
    if (
      event.target.closest('#remove-img-btn') ||
      event.target.closest('#selected-dish-card') ||
      event.target.closest('#select-dish-prompt')
    ) {
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

  // Reopen the picker from the "Select dish from the image" card
  selectDishPrompt.addEventListener('click', (event) => {
    event.stopPropagation();
    if (detectedMatches.length >= 2) {
      showDishPicker(detectedMatches);
    }
  });

  // Escape closes the picker (the prompt card then appears)
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !picker.classList.contains('hidden')) {
      closeDishPicker();
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
    setOrigin('');
    previewState.style.display = 'none';
    defaultState.style.display = 'block';
  });

  imageForm.addEventListener('submit', (event) => {
    // The origin select is hidden now, so the browser can't point at it.
    // Check it here and show our own message instead.
    if (!fileInput.files[0]) {
      event.preventDefault();
      showErrorModal('Upload a Photo', 'Upload a photo of a menu first so we can find the dish and its country.');
      return;
    }
    if (!hiddenDishInput.value || !originSelect.value) {
      event.preventDefault();
      showErrorModal('Choose a Dish', 'Select one of the detected menu dishes before starting the comparison.');
    }
  });
});