#include <stdlib.h>
#define COMMENT "Corner_Detector_GPU"
#define RGB_COMPONENT_COLOR 255


typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *) malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
				filename);
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;
	img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

void writePPM(const char *filename, PPMImage *img) {
	FILE *fp;
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	fprintf(fp, "P6\n");
	fprintf(fp, "# %s\n", COMMENT);
	fprintf(fp, "%d %d\n", img->x, img->y);
	fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

	fwrite(img->data, 3 * img->x, img->y, fp);
	fclose(fp);
}

PPMImage* createImage(int rows , int cols){
	PPMImage * img = (PPMImage *)malloc(sizeof(PPMImage));
	img->y = rows; img->x = cols;
	img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));
	return img;
}

void freeImage(PPMImage* img){
	if (img){
		free(img->data);
		free(img);
	}
}
