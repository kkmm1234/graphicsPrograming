using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class addCude : MonoBehaviour
{
    public GameObject cube;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        addCube();
    }

    void addCube()
    {
        //get key down so only one cube is added
        if(Input.GetKeyDown(KeyCode.Space))
        {
            //create a cube
            Instantiate(cube, new Vector3(5, 5, 0), Quaternion.identity);
            Debug.Log("Cube Added");
        }
    }
}
