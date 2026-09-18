// Global Human Modification v3 historical overall modification (1990-2020)
// Source: Awesome GEE Community Catalog
// GEE asset: projects/sat-io/open-datasets/GHM/HM_1990_2020_OVERALL_300M
// Historical years: 1990, 1995, 2000, 2005, 2010, 2015, 2020

var ghm = ee.ImageCollection(
  'projects/sat-io/open-datasets/GHM/HM_1990_2020_OVERALL_300M'
);
var years = [1990, 1995, 2000, 2005, 2010, 2015, 2020];
var vis = {
  min: 0,
  max: 1,
  palette: ['4c6100', 'adda25', 'e2ff9b', 'ffff73', 'ffe629', 'ffd37f', 'ffaa00', 'e69808', 'e60000', 'a80000', '730000']
};

print('GHM-v3 historical collection', ghm);

var hm1990 = ee.Image(ghm.filter(ee.Filter.eq('year', 1990)).first());
var hm2020 = ee.Image(ghm.filter(ee.Filter.eq('year', 2020)).first());
Map.addLayer(hm1990, vis, 'GHM-v3 1990');
Map.addLayer(hm2020, vis, 'GHM-v3 2020', false);
Map.setCenter(0, 20, 2);

// Export all seven historical snapshots.
years.forEach(function(year) {
  var image = ee.Image(ghm.filter(ee.Filter.eq('year', year)).first())
    .rename('human_modification')
    .set('year', year);

  Export.image.toDrive({
    image: image,
    description: 'GHM_v3_' + year + '_300m',
    folder: 'ENM_TEST_GHM_v3_1990_2020',
    fileNamePrefix: 'GHM_v3_' + year + '_300m',
    region: image.geometry().bounds(),
    scale: 300,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
});

// Additional predictor: net human-modification change from 1990 to 2020.
var change1990to2020 = hm2020.subtract(hm1990)
  .rename('human_modification_change_1990_2020');

Map.addLayer(
  change1990to2020,
  {min: -0.5, max: 0.5, palette: ['0000ff', 'ffffff', 'ff0000']},
  'GHM change 1990-2020',
  false
);

Export.image.toDrive({
  image: change1990to2020,
  description: 'GHM_v3_change_1990_2020_300m',
  folder: 'ENM_TEST_GHM_v3_1990_2020',
  fileNamePrefix: 'GHM_v3_change_1990_2020_300m',
  region: hm2020.geometry().bounds(),
  scale: 300,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
