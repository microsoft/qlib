export declare const DEFAULT_FONT_SIZE = 12;
export declare const DEFAULT_FONT_FAMILY = "sans-serif";
export declare const DEFAULT_FONT: string;
interface Platform {
    createCanvas(): HTMLCanvasElement;
    measureText(text: string, font?: string): {
        width: number;
    };
    loadImage(src: string, onload: () => void | HTMLImageElement['onload'], onerror: () => void | HTMLImageElement['onerror']): HTMLImageElement;
}
export declare const DEFAULT_TEXT_WIDTH_MAP: Record<string, number>;
export declare const platformApi: Platform;
export declare function setPlatformAPI(newPlatformApis: Partial<Platform>): void;
export {};
