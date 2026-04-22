export declare function transformLocalCoord(out: number[], elFrom: HTMLElement, elTarget: HTMLElement, inX: number, inY: number): boolean;
export declare function transformCoordWithViewport(out: number[], el: HTMLElement, inX: number, inY: number, inverse?: boolean): boolean;
export declare function isCanvasEl(el: HTMLElement): el is HTMLCanvasElement;
export declare function encodeHTML(source: string): string;
