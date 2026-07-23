import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import { GeoJSON, GeoProjection } from './geoTypes.js';
import type Element from 'zrender/lib/Element.js';
import type { RegionOption } from './GeoModel.js';
export declare abstract class Region {
    readonly name: string;
    readonly type: 'geoJSON' | 'geoSVG';
    protected _center: number[];
    protected _rect: BoundingRect;
    constructor(name: string);
    setCenter(center: number[]): void;
    /**
     * Get center point in data unit. That is,
     * for GeoJSONRegion, the unit is lat/lng,
     * for GeoSVGRegion, the unit is SVG local coord.
     */
    getCenter(): number[];
    abstract calcCenter(): number[];
}
export declare class GeoJSONPolygonGeometry {
    readonly type = "polygon";
    exterior: number[][];
    interiors?: number[][][];
    constructor(exterior: number[][], interiors: number[][][]);
}
export declare class GeoJSONLineStringGeometry {
    readonly type = "linestring";
    points: number[][][];
    constructor(points: number[][][]);
}
export declare class GeoJSONRegion extends Region {
    readonly type = "geoJSON";
    readonly geometries: (GeoJSONPolygonGeometry | GeoJSONLineStringGeometry)[];
    properties: GeoJSON['features'][0]['properties'] & {
        echartsStyle?: Omit<RegionOption, 'name'>;
    };
    constructor(name: string, geometries: GeoJSONRegion['geometries'], cp: GeoJSON['features'][0]['properties']['cp']);
    calcCenter(): number[];
    getBoundingRect(projection?: GeoProjection): BoundingRect;
    contain(coord: number[]): boolean;
    /**
     * Transform the raw coords to target bounding.
     * @param x
     * @param y
     * @param width
     * @param height
     */
    transformTo(x: number, y: number, width: number, height: number): void;
    cloneShallow(name: string): GeoJSONRegion;
}
export declare class GeoSVGRegion extends Region {
    readonly type = "geoSVG";
    private _elOnlyForCalculate;
    constructor(name: string, elOnlyForCalculate: Element);
    calcCenter(): number[];
}
