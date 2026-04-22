import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import { GeoJSONRegion } from './Region.js';
import { GeoJSON, GeoJSONCompressed, GeoJSONSourceInput, GeoResource, GeoSpecialAreas, NameMap } from './geoTypes.js';
export declare class GeoJSONResource implements GeoResource {
    readonly type = "geoJSON";
    private _geoJSON;
    private _specialAreas;
    private _mapName;
    private _parsedMap;
    constructor(mapName: string, geoJSON: GeoJSONSourceInput, specialAreas: GeoSpecialAreas);
    /**
     * @param nameMap can be null/undefined
     * @param nameProperty can be null/undefined
     */
    load(nameMap: NameMap, nameProperty: string): {
        regions: GeoJSONRegion[];
        boundingRect: BoundingRect;
        regionsMap: import("zrender/lib/core/util").HashMap<GeoJSONRegion, string | number>;
    };
    private _parseToRegions;
    /**
     * Only for exporting to users.
     * **MUST NOT** used internally.
     */
    getMapForUser(): {
        geoJson: GeoJSON | GeoJSONCompressed;
        geoJSON: GeoJSON | GeoJSONCompressed;
        specialAreas: GeoSpecialAreas;
    };
}
