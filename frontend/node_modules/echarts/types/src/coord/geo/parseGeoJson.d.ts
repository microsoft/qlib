import { GeoJSONRegion } from './Region.js';
import { GeoJSONCompressed, GeoJSON } from './geoTypes.js';
export default function parseGeoJSON(geoJson: GeoJSON | GeoJSONCompressed, nameProperty: string): GeoJSONRegion[];
