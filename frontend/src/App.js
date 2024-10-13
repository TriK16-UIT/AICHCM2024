import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(10);
  const [searchType, setSearchType] = useState('text');
  const [model, setModel] = useState('clip');
  const [classesList, setClassesList] = useState([]);
  const [selectedObject, setSelectedObject] = useState('');
  const [quantity, setQuantity] = useState(1);
  const [classDict, setClassDict] = useState({});
  const [images, setImages] = useState([]);
  const [expandedImages, setExpandedImages] = useState([]);
  const [selectedSearchImages, setSelectedSearchImages] = useState([]);
  const [selectedExpandedImages, setSelectedExpandedImages] = useState([]);
  const [expandCount, setExpandCount] = useState(3);
  const [sortType, setSortType] = useState('default');
  const [index, setIndex] = useState(null);
  const [accumulatedUnselectedIndices, setAccumulatedUnselectedIndices] = useState([]);

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = async () => {
    const response = await axios.get('http://localhost:5000/api/classes');
    setClassesList(response.data);
  };

  const handleSearch = async () => {
    const response = await axios.post('http://localhost:5000/api/search', {
      query,
      k,
      search_type: searchType,
      class_dict: classDict,
      model,
      index,
      filter_type: "including"
    });
    setImages(response.data);
    setExpandedImages([]);
    setSelectedSearchImages([]);
    setSelectedExpandedImages([]);
    setAccumulatedUnselectedIndices([]);
  };

  const handleSearchAgain = async () => {
    if (searchType === "text") {
      const unselectedImages = images.filter((_, i) => !selectedSearchImages.includes(i));
      const unselectedIndices = unselectedImages.map(img => parseInt(img.caption.match(/Index: (\d+)/)[1]));
      const newAccumulatedUnselectedIndices = [...accumulatedUnselectedIndices, ...unselectedIndices];
      setAccumulatedUnselectedIndices(newAccumulatedUnselectedIndices);

      const response = await axios.post('http://localhost:5000/api/search', {
        query,
        k,
        search_type: searchType,
        class_dict: classDict,
        model,
        index: newAccumulatedUnselectedIndices,
        filter_type: "excluding"
      });
      setImages(response.data);
      setSelectedSearchImages([]);
    } else {
      alert("Search again is currently available for Text method only");
    }
  };

  const handleExpand = async () => {
    if (selectedSearchImages.length === 1) {
      const selectedPath = images[selectedSearchImages[0]].path;
      const response = await axios.post('http://localhost:5000/api/expand', {
        selected_path: selectedPath,
        expand_count: expandCount,
      });
      setExpandedImages(response.data);
    }
  };

  const handleAddObject = () => {
    setClassDict({ ...classDict, [selectedObject]: quantity });
  };

  const handleClearObjects = () => {
    setClassDict({});
  };

  const handleDownload = async () => {
    const selectedImages = [...selectedSearchImages.map(i => images[i]), ...selectedExpandedImages.map(i => expandedImages[i])];
    const allImages = selectedImages.length > 0 ? selectedImages : images;

    const response = await axios.post('http://localhost:5000/api/download', {
      images: allImages
    }, { responseType: 'blob' });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'search_results.csv');
    document.body.appendChild(link);
    link.click();
  };

  const renderImages = (imageList, selectedImages, setSelectedImages) => {
    const sortedImages = sortType === 'video_id'
      ? Object.entries(imageList.reduce((acc, img) => {
          const videoId = img.caption.match(/Video ID: (\w+)/)[1];
          if (!acc[videoId]) acc[videoId] = [];
          acc[videoId].push(img);
          return acc;
        }, {}))
      : [['', imageList]];

    return sortedImages.map(([videoId, images]) => (
      <div key={videoId}>
        {videoId && <h3>Video ID: {videoId}</h3>}
        <div style={{ display: 'flex', flexWrap: 'wrap' }}>
          {images.map((img, index) => (
            <div key={index} style={{ margin: '10px', textAlign: 'center' }}>
              <img
                src={img.path}
                alt={`Result ${index}`}
                style={{
                  width: '200px',
                  border: selectedImages.includes(index) ? '2px solid blue' : 'none',
                }}
                onClick={() => {
                  const newSelected = selectedImages.includes(index)
                    ? selectedImages.filter(i => i !== index)
                    : [...selectedImages, index];
                  setSelectedImages(newSelected);
                }}
              />
              <p>{img.caption}</p>
            </div>
          ))}
        </div>
      </div>
    ));
  };

  return (
    <div>
      <h1>Image Retrieval System</h1>
      <div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Input query"
        />
        <input
          type="number"
          value={k}
          onChange={(e) => setK(parseInt(e.target.value))}
          placeholder="No. of images"
        />
        <input
          type="number"
          value={index}
          onChange={(e) => setIndex(e.target.value ? parseInt(e.target.value) : null)}
          placeholder="Index"
        />
        <select value={searchType} onChange={(e) => setSearchType(e.target.value)}>
          <option value="text">Text</option>
          <option value="ocr_embedding">OCR Embedding</option>
          <option value="ocr_tfidf">OCR TF-IDF</option>
          <option value="speech">Speech</option>
        </select>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="clip">CLIP</option>
          <option value="blip">BLIP</option>
        </select>
        <button onClick={handleSearch}>Search</button>
      </div>
      <div>
        <select value={selectedObject} onChange={(e) => setSelectedObject(e.target.value)}>
          {classesList.map((cls) => (
            <option key={cls} value={cls}>{cls}</option>
          ))}
        </select>
        <input
          type="number"
          value={quantity}
          onChange={(e) => setQuantity(parseInt(e.target.value))}
          min="1"
        />
        <button onClick={handleAddObject}>Add</button>
        <button onClick={handleClearObjects}>Clear</button>
      </div>
      <div>
        <h3>Selected Objects:</h3>
        {Object.entries(classDict).map(([obj, count]) => (
          <span key={obj}>{obj}: {count} </span>
        ))}
      </div>
      <div>
        <select value={sortType} onChange={(e) => setSortType(e.target.value)}>
          <option value="default">Default</option>
          <option value="video_id">Video ID</option>
        </select>
      </div>
      <h2>Search Results</h2>
      {renderImages(images, selectedSearchImages, setSelectedSearchImages)}
      <div>
        <button onClick={handleSearchAgain}>Search Again with Feedback</button>
        <input
          type="number"
          value={expandCount}
          onChange={(e) => setExpandCount(parseInt(e.target.value))}
          min="1"
        />
        <button onClick={handleExpand}>Expand</button>
      </div>
      <h2>Expanded Images</h2>
      {renderImages(expandedImages, selectedExpandedImages, setSelectedExpandedImages)}
      <div>
        <button onClick={handleDownload}>
          {selectedSearchImages.length > 0 || selectedExpandedImages.length > 0
            ? "Download Selected Results as CSV"
            : "Download All Results as CSV"}
        </button>
      </div>
    </div>
  );
}

export default App;