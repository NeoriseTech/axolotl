'''Register custom sharegpt conversations'''

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

def register_custom_conversations():
    register_conv_template(
        Conversation(
            name="apm",
            system_template="<classifier_prompt>{system_message}",
            system_message="You are a helpful assistant tasked with labelling english messages",
            roles=["<classifier_input>", "<classifier_label>"],
            sep_style=SeparatorStyle.NO_COLON_TWO,
            sep="",
            sep2="</s>",
        )
    )
