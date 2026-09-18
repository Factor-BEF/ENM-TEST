// Black spruce (Picea mariana) historical dominant-species presence
// Source: Canada Long-Term Tree Species 1984-2022, Awesome GEE Community Catalog
// GEE asset: projects/sat-io/open-datasets/CA_FOREST/SPECIES-1984-2022
// Black spruce class value: 18

var speciesTS = ee.ImageCollection(
  'projects/sat-io/open-datasets/CA_FOREST/SPECIES-1984-2022'
);

var BLACK_SPRUCE = 18;
var years = [];
for (var y = 1984; y <= 2022; y++) years.push(y);

print('Canada tree-species collection', speciesTS);

// Sanity-check two endpoints in the map.
var img1984 = ee.Image(speciesTS.filterDate('1984-01-01', '1985-01-01').first());
var img2022 = ee.Image(speciesTS.filterDate('2022-01-01', '2023-01-01').first());
Map.addLayer(img1984.eq(BLACK_SPRUCE).selfMask(), {palette: ['006400']}, 'Picea mariana 1984');
Map.addLayer(img2022.eq(BLACK_SPRUCE).selfMask(), {palette: ['00a000']}, 'Picea mariana 2022', false);
Map.centerObject(img2022.geometry(), 3);

// Creates 39 export tasks (1984-2022). Run the tasks in the GEE Tasks panel.
years.forEach(function(year) {
  var start = year + '-01-01';
  var end = (year + 1) + '-01-01';
  var source = ee.Image(speciesTS.filterDate(start, end).first()).select([0]);

  // 1 = black spruce is the dominant tree-species class; 0 = another class/non-tree.
  var presence = source.eq(BLACK_SPRUCE)
    .toByte()
    .rename('Picea_mariana_presence')
    .set('year', year)
    .set('species', 'Picea mariana')
    .set('source_class_value', BLACK_SPRUCE);

  Export.image.toDrive({
    image: presence,
    description: 'Picea_mariana_' + year,
    folder: 'ENM_TEST_black_spruce_1984_2022',
    fileNamePrefix: 'Picea_mariana_' + year,
    region: source.geometry().bounds(),
    scale: 30,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
});
