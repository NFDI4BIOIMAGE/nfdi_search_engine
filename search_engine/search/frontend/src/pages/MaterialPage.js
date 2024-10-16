import React, { useEffect, useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../assets/styles/style.css';
import bgSearchbar from '../assets/images/bg-searchbar.jpg';
import FilterCard from '../components/FilterCard';
import ResultsBox from '../components/ResultsBox';
import Pagination from '../components/Pagination';
import PagesSelection from '../components/PagesSelection';
import { Spinner } from 'react-bootstrap';

const MaterialPage = () => {
  const [materials, setMaterials] = useState([]);
  const [facets, setFacets] = useState({});
  const [selectedFilters, setSelectedFilters] = useState({});
  const [hasLoaded, setHasLoaded] = useState(false);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10); // Default items per page

  useEffect(() => {
    const savedFilters = JSON.parse(localStorage.getItem('selectedFilters'));
    if (savedFilters) {
      setSelectedFilters(savedFilters);
    }

    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/materials', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        const uniqueMaterialsMap = new Map();
        data.forEach(material => {
          if (material.url && !uniqueMaterialsMap.has(material.url)) {
            uniqueMaterialsMap.set(material.url, material);
          }
        });

        const uniqueMaterials = Array.from(uniqueMaterialsMap.values());

        setMaterials(uniqueMaterials);
        generateFacets(uniqueMaterials);
        setHasLoaded(true);
      } catch (error) {
        console.error("Error fetching the materials data:", error);
        setError("An error occurred while fetching materials. Please try again later.");
        setHasLoaded(true);
      }
    };

    fetchData();
  }, []);

  const generateFacets = (data) => {
    const authors = {};
    const licenses = {};
    const types = {};
    const tags = {};

    data.forEach((item) => {
      // Ensure authors is an array before calling forEach
      if (Array.isArray(item.authors)) {
        item.authors.forEach(author => {
          authors[author] = (authors[author] || 0) + 1;
        });
      }

      // Ensure license is treated as an array
      if (item.license) {
        const licenseArray = Array.isArray(item.license) ? item.license : [item.license];
        licenseArray.forEach(license => {
          licenses[license] = (licenses[license] || 0) + 1;
        });
      }

      // Ensure type is treated as an array
      if (item.type) {
        const typeArray = Array.isArray(item.type) ? item.type : [item.type];
        typeArray.forEach(type => {
          types[type] = (types[type] || 0) + 1;
        });
      }

      // Ensure tags is an array before calling forEach
      if (Array.isArray(item.tags)) {
        item.tags.forEach(tag => {
          tags[tag] = (tags[tag] || 0) + 1;
        });
      }
    });

    setFacets({
      authors: Object.keys(authors).map(key => ({ key, doc_count: authors[key] })),
      licenses: Object.keys(licenses).map(key => ({ key, doc_count: licenses[key] })),
      types: Object.keys(types).map(key => ({ key, doc_count: types[key] })),
      tags: Object.keys(tags).map(key => ({ key, doc_count: tags[key] })),
    });
  };

  const handleFilter = (field, value) => {
    const updatedFilters = { ...selectedFilters };
    if (updatedFilters[field]?.includes(value)) {
      updatedFilters[field] = updatedFilters[field].filter(item => item !== value);
    } else {
      updatedFilters[field] = [...(updatedFilters[field] || []), value];
    }
    setSelectedFilters(updatedFilters);
    localStorage.setItem('selectedFilters', JSON.stringify(updatedFilters));
  };

  const filteredMaterials = materials.filter(material => {
    return Object.keys(selectedFilters).every(field => {
      return selectedFilters[field]?.length === 0 || selectedFilters[field]?.some(filterValue => {
        return Array.isArray(material[field]) ? material[field].includes(filterValue) : material[field] === filterValue;
      });
    });
  });

  const highlightFields = Object.values(selectedFilters).flat();

  // Pagination logic
  const indexOfLastMaterial = currentPage * itemsPerPage;
  const indexOfFirstMaterial = indexOfLastMaterial - itemsPerPage;
  const currentMaterials = filteredMaterials.slice(indexOfFirstMaterial, indexOfLastMaterial);

  const totalPages = Math.ceil(filteredMaterials.length / itemsPerPage);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handleItemsPerPageChange = (numItems) => {
    setItemsPerPage(numItems);
    setCurrentPage(1); // Reset to the first page when the number of items per page changes
  };

  return (
    <div>
      {/* Header with background image */}
      <div className="container-fluid py-5 mb-5 searchbar-header" style={{ position: 'relative', backgroundImage: `url(${bgSearchbar})`, backgroundSize: 'cover', backgroundPosition: 'center' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0, 0, 0, 0.1)' }}></div>
        <div className="container py-5" style={{ position: 'relative', zIndex: 1 }}>
          <div className="row justify-content-center py-5">
            <div className="col-lg-10 pt-lg-5 mt-lg-5 text-center">
              <h1 className="display-3 text-white mb-3 animated slideInDown">Materials</h1>
            </div>
          </div>
        </div>
      </div>

      {/* Main content area */}
      <div className="container my-5">
        <div className="row">
          {/* Filter Sidebar */}
          <div className="col-md-3">
            <h3>Filter by</h3>
            {Object.keys(facets).length > 0 ? (
              <>
                <FilterCard title="Authors" items={facets.authors || []} field="authors" selectedFilters={selectedFilters} handleFilter={handleFilter} />
                <FilterCard title="Licenses" items={facets.licenses || []} field="license" selectedFilters={selectedFilters} handleFilter={handleFilter} />
                <FilterCard title="Types" items={facets.types || []} field="type" selectedFilters={selectedFilters} handleFilter={handleFilter} />
                <FilterCard title="Tags" items={facets.tags || []} field="tags" selectedFilters={selectedFilters} handleFilter={handleFilter} />
              </>
            ) : (
              <p>No filters available.</p>
            )}
          </div>

          {/* Material List */}
          <div className="col-md-9">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <p>Showing {indexOfFirstMaterial + 1} to {indexOfLastMaterial > filteredMaterials.length ? filteredMaterials.length : indexOfLastMaterial} of {filteredMaterials.length} materials</p>
              {/* Items Per Page Dropdown */}
              <PagesSelection
                itemsPerPage={itemsPerPage}
                onItemsPerPageChange={handleItemsPerPageChange}
              />
            </div>

            {hasLoaded ? (
              error ? (
                <p className="text-danger">{error}</p>
              ) : (
                <>
                  <div className="materials-list">
                    {currentMaterials.length > 0 ? (
                      currentMaterials.map((material, index) => (
                        <ResultsBox
                          key={index}
                          title={material.name}
                          url={material.url}
                          authors={material.authors}
                          description={material.description}
                          license={material.license}
                          type={material.type}
                          tags={material.tags}
                          highlights={highlightFields}
                        />
                      ))
                    ) : (
                      <p>No materials found with the current filters.</p>
                    )}
                  </div>

                  {/* Pagination Controls */}
                  <Pagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    onPageChange={handlePageChange}
                  />
                </>
              )
            ) : (
              <div className="text-center">
                <Spinner animation="border" variant="primary" />
                <p>Loading materials...</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MaterialPage;
