import pickle
import streamlit as st
import numpy as np
from PIL import Image





model=pickle.load(open("model.pkl","rb"))

def main():
    st.title("⚡ Power Grid Stability Prediction")

    image=Image.open("Stability Prediction.png")
    st.image(image,use_container_width=True)



    st.header("⏱️Time Constants (τ)")
    st.caption("Represents how quickly each node responds to changes in the grid")

    tau1=st.slider("Node 1 Time Constant (τ1)",0.0,10.0,1.0)
    tau2=st.slider("Node 2 Time Constant (τ2)",0.0,10.0,1.0)
    tau3=st.slider("Node 3 Time Constant(τ3)",0.0,10.0,1.0)
    tau4=st.slider("Node 4 Time Constant(τ4)",0.0,10.0,1.0)

    st.header("⚡Power Injection / Consumption (p)")

    p1=st.slider("Node 1 Power (p1)",-2.0,2.0,0.0)
    p2=st.slider("Node 2 Power (p2)",-2.0,2.0,0.0)
    p3=st.slider("Node 3 Power (p3)",-2.0,2.0,0.0)
    p4=st.slider("Node 4 Power (p4)",-2.0,2.0,0.0)


    st.header("🎛️ Control Gain (g)")
    st.caption("Determines how strongly each node reacts to system imbalance")

    g1=st.slider("Node 1 Control Gain (g1)",0.0,1.0,0.5)
    g2=st.slider("Node 2 Control Gain (g2)",0.0,1.0,0.5)
    g3=st.slider("Node 3 Control Gain (g3)",0.0,1.0,0.5)
    g4=st.slider("Node 4 Control Gain (g4)",0.0,1.0,0.5)

    if st.button("Predict"):
        input_data=np.array([[tau1,tau2,tau3,tau4,p1,p2,p3,p4,g1,g2,g3,g4]])
        prediction=model.predict(input_data)[0]

        st.subheader("📊 Result")

        if prediction<0:
            st.success("Stable System ✅")
        else:
            st.error("Unstable System ⚠️")

        st.metric("Stability Score",f"{prediction:.4f}")
    st.info("Score < -0.1 → Stable | -0.1 to 0.1 → Moderate Risk | > 0.1 → Unstable")
    st.markdown("---")
    st.caption("Dataset: Electrical Grid Stability (UCI) | Model: XGBoost") 

main()























# import pickle
# import streamlit as st
# import numpy as np

# st.set_page_config(page_title="Power Grid Stability",page_icon="⚡")

# model=pickle.load(open("model.pkl","rb"))

# def input_block(label,min_val,max_val,default,key):

#     if key not in st.session_state:
#         st.session_state[key]=default
#         st.session_state[key+"_slider"]=default
#         st.session_state[key+"_input"]=default

#     def update_input():
#         val=st.session_state[key+"_slider"]
#         st.session_state[key]=val
#         st.session_state[key+"_input"]=val

#     def update_slider():
#         val=st.session_state[key+"_input"]
#         st.session_state[key]=val
#         st.session_state[key+"_slider"]=val

#     col1,col2=st.columns([3,1])

#     with col1:
#         st.slider(
#             label,
#             min_val,
#             max_val,
#             key=key+"_slider",
#             on_change=update_input
#         )

#     with col2:
#         st.number_input(
#             "Value",
#             min_val,
#             max_val,
#             key=key+"_input",
#             on_change=update_slider
#         )

#     return st.session_state[key]

# # ------------------ MAIN APP ------------------
# def main():

#     st.title("⚡ Power Grid Stability Prediction")

#     st.header("⏱️ Time Constants (τ)")
#     st.caption("Represents how quickly each node responds to changes in the grid")

#     tau1=input_block("Node 1 Time Constant (τ1)",0.0,10.0,1.0,"tau1")
#     tau2=input_block("Node 2 Time Constant (τ2)",0.0,10.0,1.0,"tau2")
#     tau3=input_block("Node 3 Time Constant (τ3)",0.0,10.0,1.0,"tau3")
#     tau4=input_block("Node 4 Time Constant (τ4)",0.0,10.0,1.0,"tau4")

#     st.header("⚡ Power Injection / Consumption (p)")
#     st.caption("Positive = generation | Negative = consumption")

#     p1=input_block("Node 1 Power (p1)",-2.0,2.0,0.0,"p1")
#     p2=input_block("Node 2 Power (p2)",-2.0,2.0,0.0,"p2")
#     p3=input_block("Node 3 Power (p3)",-2.0,2.0,0.0,"p3")
#     p4=input_block("Node 4 Power (p4)",-2.0,2.0,0.0,"p4")

#     st.header("🎛️ Control Gain (g)")
#     st.caption("Determines how strongly each node reacts to system imbalance")

#     g1=input_block("Node 1 Control Gain (g1)",0.0,1.0,0.5,"g1")
#     g2=input_block("Node 2 Control Gain (g2)",0.0,1.0,0.5,"g2")
#     g3=input_block("Node 3 Control Gain (g3)",0.0,1.0,0.5,"g3")
#     g4=input_block("Node 4 Control Gain (g4)",0.0,1.0,0.5,"g4")

#     if st.button("🚀 Predict"):

#         input_data=np.array([[tau1,tau2,tau3,tau4,p1,p2,p3,p4,g1,g2,g3,g4]])
#         prediction=model.predict(input_data)[0]

#         st.subheader("📊 Result")

#         if prediction<0:
#             st.success("Stable System ✅")
#         else:
#             st.error("Unstable System ⚠️")

#         st.metric("Stability Score",f"{prediction:.4f}")

#     st.info("Negative score → Stable | Positive score → Unstable")
#     st.markdown("---")
#     st.caption("Dataset: Electrical Grid Stability (UCI) | Model: XGBoost")

# main()
