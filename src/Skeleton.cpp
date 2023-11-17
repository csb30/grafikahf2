//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Cseh Balint Istvan
// Neptun : WRNJPE
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include <iostream>
#include "framework.h"

int current_time=1;
std::vector<float> star, red, green, blue;
float max=0;
bool max_calculated = false;

//raytrace.cpp
struct Material {
    vec3 ka, kd, ks; //ambiens, diffuz, spekularis
    float  shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
    float t;
    vec3 position, normal; //metszes pozija, normalvektora
    Material * material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir; //kezdopont, iranyvektor
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
    Material * material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
    virtual void move(int t) = 0;
};

struct Sphere : public Intersectable {
    vec3 center;
    float radius;

    Sphere(const vec3& _center, float _radius, Material* _material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) * (1.0f / radius);
        hit.material = material;
        return hit;
    }

    void move(int t){
        if(t-current_time==0) return;
        //Hubble
        for(int i=0; i<abs(t-current_time); i++){
            if (t-current_time>0){
                center=center + 0.1*center;
            }
            else {
                center=center - 0.1*center;
            }
        }
    }
};

class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) { //kamera pont, hova nez, merre van a felfele, radianban nyilasszog
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f; //kerekites/korrekcio

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La; //ambiens feny
public:
    void build() {
        float fov = 4.0f * M_PI / 180;
        vec3 eye = vec3(0, 0, -0.5f/(tan(fov / 2.0f))), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        //eye.z = 1/tan(fov/2)
        camera.set(eye, lookat, vup, fov);

        La = vec3(1.0f, 1.0f, 1.0f);
        //vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
        //lights.push_back(new Light(lightDirection, Le));


        vec3 kd(1.0f, 1.0f, 1.0f), ks(2, 2, 2);
        Material *material = new Material(kd, ks, 50);
        //70-400
        for (int i = 0; i < 100; i++) {
            float z = rnd()*330 + 70;
            float x = (rnd()*2-1)*tan(fov/2)*z;
            float y = (rnd()*2-1)*tan(fov/2)*z;
            objects.push_back(new Sphere(vec3(x, y, z), 0.5f, material));
        }
    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return vec3(0,0,0);
        vec3 hermite_color = (0,0,0);
        //color
        float c = 299792.458 * 60 * 60; //km/h
        float v = 0.1f*hit.t * 1000 / 365 / 24;
        for(int i=400; i<=700; i++){
            int idx=i-400;
            int wave = round(float(i) * (c+v) / c);
            hermite_color.x += star[wave-250]*red[idx];
            hermite_color.y += star[wave-250]*green[idx];
            hermite_color.z += star[wave-250]*blue[idx];
        }

        //normalize
        if (!max_calculated) {
            max = hermite_color.x;
            if (max < hermite_color.y) max = hermite_color.y;
            if (max < hermite_color.z) max = hermite_color.z;
            max_calculated = true;
        }
        hermite_color.x = 5.0f * hermite_color.x / max;
        hermite_color.y = 5.0f * hermite_color.y / max;
        hermite_color.z = 5.0f * hermite_color.z / max;

        //vec3 outRadiance = hit.material->ka * La;
        vec3 outRadiance = hermite_color * La * (1-(hit.t-70)/660);
        for (Light * light : lights) {
            Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
            float cosTheta = dot(hit.normal, light->direction);
            if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
                outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                vec3 halfway = normalize(-ray.dir + light->direction);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
            }
        }
        return outRadiance;
    }

    void moveObjects(int t){
        for(int i = 0; i < objects.size(); i++){
            objects[i]->move(t);
        }
        current_time=t;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
    int windowWidth, windowHeight;
    std::vector<vec4> image;
    Texture texture;

public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight):
    windowWidth(windowWidth), windowHeight(windowWidth), image(std::vector<vec4>(windowWidth*windowHeight)),
    texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void renderScreen(){
        scene.render(image);
        texture.create(windowWidth, windowHeight, image);
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
    }
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

float hermite(float p0, float p1, float t0, float t1, float t){
    float a3 = (2.0f*(p0-p1)) / pow(t1-t0,3);
    float a2 = (3.0f*(p1-p0)) / pow(t1-t0, 2);
    float a0 = p0;
    return a3*pow(t-t0,3) + a2*pow(t-t0,2) + a0;
}


// Initialization, create an OpenGL context
void onInitialization() {

    //150-1600

    //star
    for (int i=150; i<450; i++){
        star.push_back(hermite(0,1,150,450,float(i)));
    }
    for (int i=450; i<=1600; i++){
        star.push_back(hermite(1,0.1,450,1600,float(i)));
    }

    //400 - 700

    //red
    for(int i=400; i<500; i++){
        red.push_back(hermite(0,-0.2,400,500,float(i)));
    }
    for(int i=500; i<600; i++){
        red.push_back(hermite(-0.2,2.5,500,600,float(i)));
    }
    for(int i=600; i<=700; i++){
        red.push_back(hermite(2.5,0,600,700,float(i)));
    }

    //green
    for(int i=400; i<500; i++){
        green.push_back(hermite(0,-0.1,400,500,float(i)));
    }
    for(int i=500; i<600; i++){
        green.push_back(hermite(-0.1,1.2,500,600,float(i)));
    }
    for(int i=600; i<=700; i++){
        green.push_back(hermite(1.2,0,600,700,float(i)));
    }

    //blue
    for(int i=400; i<460; i++){
        blue.push_back(hermite(0,1,400,460,float(i)));
    }
    for(int i=460; i<520; i++){
        blue.push_back(hermite(1,0,460,520,float(i)));
    }
    for(int i=520; i<=700; i++){
        blue.push_back(0);
    }

    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    fullScreenTexturedQuad->renderScreen();
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();									// exchange the two buffers
}


//skeleton.cpp


// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    bool modified = false;
    switch(key){
        case '0':
            scene.moveObjects(2);
            modified=true;
            break;
        case '1':
            scene.moveObjects(4);
            modified=true;
            break;
        case '2':
            scene.moveObjects(6);
            modified=true;
            break;
        case '3':
            scene.moveObjects(8);
            modified=true;
            break;
        case '4':
            scene.moveObjects(10);
            modified=true;
            break;
        case '5':
            scene.moveObjects(12);
            modified=true;
            break;
        case '6':
            scene.moveObjects(14);
            modified=true;
            break;
        case '7':
            scene.moveObjects(16);
            modified=true;
            break;
        case '8':
            scene.moveObjects(18);
            modified=true;
            break;
        case '9':
            scene.moveObjects(20);
            modified=true;
            break;
    }
    if(modified) {
        fullScreenTexturedQuad->renderScreen();
        glutPostRedisplay();
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
